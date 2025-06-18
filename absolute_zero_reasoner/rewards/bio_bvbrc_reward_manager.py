import os
import re
import uuid
import json
from functools import partial
from typing import Dict, Any, List, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from verl import DataProto
from verl.protocol import DataProtoItem

from absolute_zero_reasoner.rewards.custom_evaluate import get_format_reward, extract_answer, extract_thought
from absolute_zero_reasoner.utils.logging_utils.stdout import PrettyPrinter
from absolute_zero_reasoner.utils.pseudo_chain_processor import create_execution_only_processor


class BioReasoningRewardManager:
    """
    Reward manager for biological reasoning tasks using BV-BRC.
    Integrates pseudo-chain processing for execution-based reasoning.
    """
    
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        num_examine: int,
        split: str,
        reward_fn_extraction_type: str,
        splitter: str,
        output_path: str,
        debug: bool = False,
        max_prompt_length: int = 8192,
        bvbrc_timeout: int = 30,
        enable_pseudo_chain: bool = True,
        max_fix_iterations: int = 3,
        boxed_retry: bool = False,
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.split = split
        self.reward_fn_extraction_type = reward_fn_extraction_type
        self.splitter = splitter
        self.output_path = output_path
        self.max_prompt_length = max_prompt_length
        self.debug = debug
        self.boxed_retry = boxed_retry
        
        # Pseudo-chain processing
        self.enable_pseudo_chain = enable_pseudo_chain
        self.max_fix_iterations = max_fix_iterations
        if self.enable_pseudo_chain:
            self.pseudo_chain_processor = create_execution_only_processor(
                bvbrc_timeout=bvbrc_timeout
            )
        else:
            self.pseudo_chain_processor = None
    
    def _get_data_dict(self, data_item: DataProtoItem, uid: str) -> Dict:
        """Extract data dictionary from a data item"""
        prompt_ids = data_item.batch['prompts']
        prompt_length = prompt_ids.shape[-1]
        valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        response_ids = data_item.batch['responses']
        valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        # Decode sequences
        sequences = torch.cat((valid_prompt_ids, valid_response_ids))
        sequences_str = self.tokenizer.decode(sequences)
        
        ground_truth = data_item.non_tensor_batch.get('ground_truth', '')
        data_source = data_item.non_tensor_batch.get('data_source', 'unknown')
        extra_info = data_item.non_tensor_batch.get('extra_info', {})
        user_query = data_item.non_tensor_batch.get('problem', '')
        
        non_special_tokens_sequences_str = self.tokenizer.decode(
            self.tokenizer.encode(sequences_str), skip_special_tokens=True
        )
        
        # Extract the generation part
        generation = non_special_tokens_sequences_str.split(self.splitter)[1].strip().strip('\"\'')
        
        # Process with pseudo-chain if enabled
        if self.enable_pseudo_chain and self.pseudo_chain_processor:
            try:
                processed_generation = self.pseudo_chain_processor.process_response(
                    generation, user_query, max_iterations=self.max_fix_iterations
                )
            except Exception as e:
                PrettyPrinter.status("PSEUDO_CHAIN", f"Processing failed: {str(e)}", "warning")
                processed_generation = generation
        else:
            processed_generation = generation
        
        # Extract content and thought
        extracted_content = extract_answer(
            processed_generation, 
            self.reward_fn_extraction_type, 
            boxed_retry=self.boxed_retry
        )
        thought = extract_thought(processed_generation)
        
        data_dict = {
            'generation': generation,
            'processed_generation': processed_generation,
            'data_source': data_source,
            'ground_truth': ground_truth,
            'extra_info': extra_info,
            'user_query': user_query,
            'non_special_tokens_sequences_str': non_special_tokens_sequences_str,
            'valid_response_length': valid_response_length,
            'extracted_content': extracted_content,
            'thought': thought,
            'uid': uid,
        }
        
        return data_dict
    
    def _compute_bio_reasoning_reward(self, data_dict: Dict) -> Tuple[float, Dict]:
        """
        Compute reward for biological reasoning based on execution results.
        
        Returns:
            reward: Float reward value
            metrics: Dictionary of detailed metrics
        """
        processed_generation = data_dict['processed_generation']
        user_query = data_dict['user_query']
        
        # Initialize metrics
        metrics = {
            'has_actions': 0,
            'actions_executed': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'execution_success_rate': 0.0,
            'has_execution_results': 0,
            'reasoning_quality': 0.0,
        }
        
        # Check if response contains actions
        action_pattern = re.compile(r'<action>\s*(.*?)\s*</action>', re.DOTALL | re.IGNORECASE)
        action_matches = action_pattern.findall(processed_generation)
        
        if action_matches:
            metrics['has_actions'] = 1
            metrics['actions_executed'] = len(action_matches)
            
            # Check for execution results
            execution_pattern = re.compile(r'<execution_results>\s*(.*?)\s*</execution_results>', 
                                         re.DOTALL | re.IGNORECASE)
            execution_matches = execution_pattern.findall(processed_generation)
            
            if execution_matches:
                metrics['has_execution_results'] = 1
                
                # Count successful vs failed executions
                for exec_result in execution_matches:
                    if 'Result:' in exec_result:
                        metrics['successful_executions'] += exec_result.count('Result:')
                    if 'Error:' in exec_result:
                        metrics['failed_executions'] += exec_result.count('Error:')
                
                total_executions = metrics['successful_executions'] + metrics['failed_executions']
                if total_executions > 0:
                    metrics['execution_success_rate'] = metrics['successful_executions'] / total_executions
        
        # Compute reasoning quality based on structure and content
        reasoning_quality = self._assess_reasoning_quality(processed_generation, user_query)
        metrics['reasoning_quality'] = reasoning_quality
        
        # Compute final reward
        reward = self._compute_final_reward(metrics, data_dict)
        
        return reward, metrics
    
    def _assess_reasoning_quality(self, processed_generation: str, user_query: str) -> float:
        """
        Assess the quality of reasoning in the processed generation.
        Returns a score between 0 and 1.
        """
        quality_score = 0.0
        
        # Check for structured reasoning
        if '<think>' in processed_generation.lower() or 'reasoning:' in processed_generation.lower():
            quality_score += 0.2
        
        # Check for proper action formatting
        if '<action>' in processed_generation and '</action>' in processed_generation:
            quality_score += 0.2
        
        #Need some other way to check quality of reasoning based on
        # 1. The successful number of executioned actions
        
        
        return min(1.0, quality_score)
    
    def _compute_final_reward(self, metrics: Dict, data_dict: Dict) -> float:
        """
        Compute the final reward based on metrics and data.
        """
        # Base format reward
        format_score = get_format_reward(
            solution_str=data_dict['processed_generation'],
            extraction_type=self.reward_fn_extraction_type
        )
        
        if format_score <= 0:
            return -1.0
        
        # Execution-based reward
        execution_reward = 0.0
        if metrics['has_actions']:
            if metrics['has_execution_results']:
                # Get actual results from execution
                actual_results = self._extract_execution_results(data_dict['processed_generation'])
                
                # Compare with ground truth if available
                expected_answer = data_dict.get('ground_truth')
                if expected_answer is not None:
                    answer_similarity = self._compute_soft_answer_similarity(actual_results, expected_answer)
                    execution_reward = answer_similarity * 0.8  # Weight answer similarity highly
                    
                    # Small bonus for successful execution
                    if metrics['successful_executions'] > 0:
                        execution_reward += 0.2
                else:
                    # Fall back to execution success rate if no ground truth
                    execution_reward = metrics['execution_success_rate'] * 0.5
                    
                    # Bonus for having any successful executions
                    if metrics['successful_executions'] > 0:
                        execution_reward += 0.3
            else:
                # Penalty for having actions but no execution results
                execution_reward = -0.2
        
        # Reasoning quality reward (simplified since we focus on answer accuracy)
        reasoning_reward = metrics['reasoning_quality'] * 0.1
        
        # Combine rewards
        if self.split == 'train':
            final_reward = execution_reward + reasoning_reward
            
            # Apply format penalty if needed
            if format_score < 1.0:
                final_reward *= format_score
                
            # Ensure reward is in reasonable range
            final_reward = max(-1.0, min(1.0, final_reward))
            
        else:  # test split
            # For evaluation, focus on answer accuracy
            expected_answer = data_dict.get('ground_truth')
            if expected_answer is not None:
                actual_results = self._extract_execution_results(data_dict['processed_generation'])
                answer_similarity = self._compute_soft_answer_similarity(actual_results, expected_answer)
                final_reward = 1.0 if answer_similarity > 0.5 else 0.0
            else:
                final_reward = 1.0 if metrics['execution_success_rate'] > 0.5 else 0.0
        
        return final_reward
    
    def _extract_execution_results(self, processed_generation: str) -> Any:
        """Extract actual results from execution output"""
        execution_pattern = re.compile(r'<execution_results>\s*(.*?)\s*</execution_results>', 
                                     re.DOTALL | re.IGNORECASE)
        execution_matches = execution_pattern.findall(processed_generation)
        
        if not execution_matches:
            return None
            
        # Look for successful results (lines starting with "Result:")
        results = []
        for exec_block in execution_matches:
            result_lines = re.findall(r'Result:\s*(.*?)(?=\n|$)', exec_block, re.DOTALL)
            for result_line in result_lines:
                try:
                    # Try to parse as JSON
                    parsed_result = json.loads(result_line.strip())
                    results.append(parsed_result)
                except json.JSONDecodeError:
                    # If not JSON, keep as string
                    results.append(result_line.strip())
        
        # Return the last successful result, or all results if multiple
        if len(results) == 1:
            return results[0]
        elif len(results) > 1:
            return results
        else:
            return None
    
    def _compute_soft_answer_similarity(self, actual: Any, expected: Any) -> float:
        """
        Compute soft similarity between actual and expected answers.
        Handles different limits, partial overlaps, and flexible formats.
        """
        if actual is None:
            return 0.0
            
        # Handle different data types
        if isinstance(expected, list) and isinstance(actual, list):
            return self._compute_list_similarity(actual, expected)
        elif isinstance(expected, dict) and isinstance(actual, dict):
            return self._compute_dict_similarity(actual, expected)
        elif isinstance(expected, str) and isinstance(actual, str):
            return self._compute_string_similarity(actual, expected)
        elif isinstance(expected, list) and isinstance(actual, dict):
            # Sometimes API returns single item as dict instead of list
            return self._compute_list_similarity([actual], expected)
        elif isinstance(expected, dict) and isinstance(actual, list):
            # Sometimes we expect dict but get list of dicts
            if actual and isinstance(actual[0], dict):
                return self._compute_dict_similarity(actual[0], expected)
        else:
            # Try string comparison as fallback
            return self._compute_string_similarity(str(actual), str(expected))
    
    def _compute_list_similarity(self, actual: List, expected: List) -> float:
        """
        Compute similarity between two lists with soft matching.
        Rewards partial overlap and doesn't penalize for different limits.
        """
        if not expected:
            return 1.0 if not actual else 0.0
        if not actual:
            return 0.0
            
        # Convert to sets for overlap calculation
        if all(isinstance(item, (str, int, float)) for item in expected):
            # Simple values - direct set comparison
            actual_set = set(str(item) for item in actual if isinstance(item, (str, int, float)))
            expected_set = set(str(item) for item in expected)
            
            overlap = len(actual_set & expected_set)
            
            # Reward based on how much of the expected we found
            # Don't penalize for finding more than expected
            similarity = overlap / len(expected_set)
            
            # Bonus if we found a good portion of the start of the list
            if len(actual) >= 3 and len(expected) >= 3:
                start_overlap = sum(1 for a, e in zip(actual[:3], expected[:3]) if str(a) == str(e))
                start_bonus = (start_overlap / 3) * 0.2
                similarity += start_bonus
                
            return min(1.0, similarity)
        else:
            # Complex objects - compare individual items
            total_similarity = 0.0
            matches = 0
            
            for expected_item in expected:
                best_match = 0.0
                for actual_item in actual:
                    if isinstance(expected_item, dict) and isinstance(actual_item, dict):
                        item_sim = self._compute_dict_similarity(actual_item, expected_item)
                        best_match = max(best_match, item_sim)
                    else:
                        item_sim = 1.0 if str(actual_item) == str(expected_item) else 0.0
                        best_match = max(best_match, item_sim)
                
                total_similarity += best_match
                if best_match > 0.5:
                    matches += 1
            
            # Average similarity across expected items
            avg_similarity = total_similarity / len(expected)
            
            # Bonus for finding at least some matches
            if matches > 0:
                match_bonus = min(0.3, matches / len(expected) * 0.3)
                avg_similarity += match_bonus
                
            return min(1.0, avg_similarity)
    
    def _compute_dict_similarity(self, actual: Dict, expected: Dict) -> float:
        """Compute similarity between two dictionaries"""
        if not expected:
            return 1.0 if not actual else 0.0
        if not actual:
            return 0.0
            
        total_score = 0.0
        for key, expected_value in expected.items():
            if key in actual:
                if isinstance(expected_value, (list, dict)):
                    # Recursive comparison for nested structures
                    total_score += self._compute_soft_answer_similarity(actual[key], expected_value)
                else:
                    # Simple value comparison
                    total_score += 1.0 if str(actual[key]) == str(expected_value) else 0.0
            # Don't penalize for missing keys too harshly - might be different API response format
        
        # Normalize by number of expected keys
        return total_score / len(expected)
    
    def _compute_string_similarity(self, actual: str, expected: str) -> float:
        """Compute similarity between two strings"""
        actual_lower = actual.lower().strip()
        expected_lower = expected.lower().strip()
        
        if expected_lower == actual_lower:
            return 1.0
        elif expected_lower in actual_lower:
            return 0.8
        elif actual_lower in expected_lower:
            return 0.6
        else:
            # Check for partial word matches
            expected_words = set(expected_lower.split())
            actual_words = set(actual_lower.split())
            
            if expected_words & actual_words:
                overlap = len(expected_words & actual_words)
                return (overlap / len(expected_words)) * 0.5
            else:
                return 0.0
    
    def __call__(
        self,
        data: DataProto,
        problem_type: str = None,
        executor = None,
        rollout_actor_wg = None,
        banned_words: List[str] = [],
        banned_assertion_keywords: List[str] = [],
        n_samples: int = 1,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict, List[Dict], List[Dict]]:
        """
        Main reward computation function.
        
        Returns:
            reward_tensor: Tensor of rewards for each response
            all_scores: Dictionary of aggregated scores
            valid_programs: List of valid programs (empty for bio reasoning)
            correct_predictions: List of correct predictions
        """
        # Check for pre-computed rewards
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']
        
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        all_scores = defaultdict(list)
        data_dicts = []
        correct_predictions = []
        
        # Generate UIDs for tracking
        uids = np.array([str(uuid.uuid4()) for _ in range(len(data))], dtype=object)
        
        PrettyPrinter.section_header("Processing Bio Reasoning Responses")
        
        # Process each data item
        for i in range(len(data)):
            data_dict = self._get_data_dict(data[i], uids[i])
            data_dicts.append(data_dict)
            
            # Compute reward and metrics
            reward, metrics = self._compute_bio_reasoning_reward(data_dict)
            
            # Assign reward to final token
            valid_response_length = data_dict['valid_response_length']
            reward_tensor[i, valid_response_length - 1] = reward
            
            # Collect metrics
            for key, value in metrics.items():
                all_scores[key].append(value)
            
            # Track correct predictions
            if reward > 0.5:  # Threshold for "correct"
                correct_predictions.append(data_dict)
        
        # Compute aggregate scores
        aggregate_scores = {}
        for key, values in all_scores.items():
            if values:
                aggregate_scores[f'bio_reasoning/{key}'] = np.mean(values)
        
        # Add summary metrics
        aggregate_scores['bio_reasoning/avg_reward'] = reward_tensor.sum(-1).mean().item()
        aggregate_scores['bio_reasoning/success_rate'] = len(correct_predictions) / len(data)
        
        PrettyPrinter.section_header("Bio Reasoning Results Summary")
        PrettyPrinter.table(
            ["Metric", "Value"],
            [[k.replace('bio_reasoning/', ''), f"{v:.3f}"] for k, v in aggregate_scores.items()],
            title="Bio Reasoning Metrics"
        )
        
        return reward_tensor, aggregate_scores, [], correct_predictions 