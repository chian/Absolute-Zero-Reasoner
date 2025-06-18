import re
import json
import ast
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum

from absolute_zero_reasoner.utils.code_utils.bvbrc_executor import BVBRCShellExecutor


class StepType(Enum):
    REASONING = "reasoning"
    ACTION = "action"
    ANSWER = "answer"


@dataclass
class ReasoningStep:
    """Represents a single step in the reasoning chain"""
    step_type: StepType
    text: str
    start_char: int
    end_char: int
    execution_result: Optional[Dict] = None
    updated: bool = False


class ExecutionClassifier:
    """Classifies BV-BRC execution results as success or failure"""
    
    @staticmethod
    def classify_execution(stdout: str, stderr: str) -> Tuple[bool, str]:
        """
        Classify a BV-BRC execution as success or failure.
        Returns (is_success, reason).
        """
        if stderr.strip():
            return False, f"Error: {stderr.strip()}"
        if not stdout.strip():
            return False, "No output returned."
        
        # Check for common API errors
        if "undefined field" in stdout.lower():
            return False, "API error: undefined field"
        if "error" in stdout.lower() and "http" in stdout.lower():
            return False, "HTTP error in response"
            
        # Try to parse as JSON to verify valid response
        try:
            json.loads(stdout.strip())
            return True, "Command executed successfully with valid JSON response"
        except json.JSONDecodeError:
            # Non-JSON output might still be valid for some queries
            if len(stdout.strip()) > 0:
                return True, "Command executed successfully with text response"
            return False, "Invalid response format"


class BVBRCActionFixer:
    """Fixes failed BV-BRC actions using LLM assistance"""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        
    def fix_failed_action(self, action_text: str, execution_result: Dict, user_query: str) -> str:
        """
        Fix a failed BV-BRC action.
        For Phase 1, we implement basic pattern-based fixes.
        Future versions can use LLM assistance.
        """
        stdout = execution_result.get('stdout', '')
        stderr = execution_result.get('stderr', '')
        
        # Basic pattern-based fixes
        if "undefined field" in stdout.lower() or "undefined field" in stderr.lower():
            # Extract the problematic field
            field_match = re.search(r"undefined field (\w+)", stdout + stderr, re.IGNORECASE)
            if field_match:
                problematic_field = field_match.group(1)
                # Try common field name corrections
                field_corrections = {
                    'genus': 'organism',  # Sometimes genus is stored as organism
                    'species': 'organism',  # Sometimes species is part of organism
                    'strain': 'genome_name',  # Sometimes strain is in genome_name
                }
                if problematic_field in field_corrections:
                    corrected_field = field_corrections[problematic_field]
                    fixed_action = action_text.replace(problematic_field, corrected_field)
                    return fixed_action
        
        # If no specific fix available, try simplifying the query
        if "curl" in action_text and "bv-brc.org" in action_text:
            # Simplify by removing complex filters and adding limit
            if "limit(" not in action_text:
                # Add limit to prevent large responses
                action_text = action_text.replace('"', '&limit(1)"')
            
        return action_text


class SubsequentStepUpdater:
    """Updates subsequent reasoning steps based on execution results"""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        
    def should_update_subsequent_steps(self, execution_result: Dict, step_index: int, total_steps: int) -> bool:
        """
        Determine if subsequent steps should be updated based on execution result.
        For Phase 1, we use simple heuristics.
        """
        # Only update if execution was successful and we're not at the last step
        if step_index >= total_steps - 1:
            return False
            
        stdout = execution_result.get('stdout', '')
        is_success = execution_result.get('is_success', False)
        
        if not is_success:
            return False
            
        # Update if we got substantial new information
        if len(stdout.strip()) > 50:  # Arbitrary threshold
            return True
            
        return False
        
    def update_subsequent_steps(self, steps: List[ReasoningStep], trigger_index: int, 
                              execution_result: Dict, user_query: str) -> List[ReasoningStep]:
        """
        Update subsequent reasoning steps based on execution results.
        For Phase 1, we implement basic text injection.
        """
        updated_steps = steps.copy()
        
        # Find the next reasoning step after the action
        next_reasoning_index = None
        for i in range(trigger_index + 1, len(steps)):
            if steps[i].step_type == StepType.REASONING:
                next_reasoning_index = i
                break
                
        if next_reasoning_index is not None:
            # Inject execution result into the next reasoning step
            stdout = execution_result.get('stdout', '')
            if stdout.strip():
                # Summarize the output if it's too long
                if len(stdout) > 200:
                    summary = stdout[:200] + "... (truncated)"
                else:
                    summary = stdout
                    
                # Prepend the result to the reasoning step
                original_text = updated_steps[next_reasoning_index].text
                updated_text = f"Based on the query result: {summary}\n\n{original_text}"
                
                updated_steps[next_reasoning_index] = ReasoningStep(
                    step_type=updated_steps[next_reasoning_index].step_type,
                    text=updated_text,
                    start_char=updated_steps[next_reasoning_index].start_char,
                    end_char=updated_steps[next_reasoning_index].end_char,
                    updated=True
                )
                
        return updated_steps


class ExecutionOnlyProcessor:
    """
    Phase 1 implementation of pseudo-chain processing.
    Handles execution-only editing similar to solr_reasoning.
    """
    
    def __init__(self, bvbrc_executor: BVBRCShellExecutor, 
                 action_fixer: Optional[BVBRCActionFixer] = None,
                 step_updater: Optional[SubsequentStepUpdater] = None):
        self.bvbrc_executor = bvbrc_executor
        self.action_fixer = action_fixer or BVBRCActionFixer()
        self.step_updater = step_updater or SubsequentStepUpdater()
        self.classifier = ExecutionClassifier()
        
    def parse_reasoning_steps(self, response_text: str) -> List[ReasoningStep]:
        """
        Parse response text into reasoning steps.
        Identifies reasoning, action, and answer sections.
        """
        steps = []
        
        # Find action blocks
        action_pattern = re.compile(r'<action>\s*(.*?)\s*</action>', re.DOTALL | re.IGNORECASE)
        action_matches = list(action_pattern.finditer(response_text))
        
        current_pos = 0
        
        for action_match in action_matches:
            # Add reasoning text before this action
            if action_match.start() > current_pos:
                reasoning_text = response_text[current_pos:action_match.start()].strip()
                if reasoning_text:
                    steps.append(ReasoningStep(
                        step_type=StepType.REASONING,
                        text=reasoning_text,
                        start_char=current_pos,
                        end_char=action_match.start()
                    ))
            
            # Add the action
            action_content = action_match.group(1).strip()
            steps.append(ReasoningStep(
                step_type=StepType.ACTION,
                text=action_content,
                start_char=action_match.start(),
                end_char=action_match.end()
            ))
            
            current_pos = action_match.end()
        
        # Add any remaining text as reasoning/answer
        if current_pos < len(response_text):
            remaining_text = response_text[current_pos:].strip()
            if remaining_text:
                # Check if this looks like an answer section
                if any(marker in remaining_text.lower() for marker in ['answer:', 'the answer is', 'conclusion:']):
                    step_type = StepType.ANSWER
                else:
                    step_type = StepType.REASONING
                    
                steps.append(ReasoningStep(
                    step_type=step_type,
                    text=remaining_text,
                    start_char=current_pos,
                    end_char=len(response_text)
                ))
        
        return steps
        
    def extract_bvbrc_commands(self, action_text: str) -> List[str]:
        """
        Extract BV-BRC curl commands from action text.
        Handles both JSON format and plain text format.
        """
        commands = []
        
        # Try to parse as JSON first
        try:
            # Handle both single command and array of commands
            if action_text.strip().startswith('['):
                # Array format
                action_data = json.loads(action_text)
                for item in action_data:
                    if isinstance(item, dict) and 'action_input' in item:
                        commands.append(item['action_input'])
                    elif isinstance(item, str):
                        commands.append(item)
            elif action_text.strip().startswith('{'):
                # Single command format
                action_data = json.loads(action_text)
                if 'action_input' in action_data:
                    commands.append(action_data['action_input'])
            else:
                # Try to extract curl commands directly
                curl_pattern = re.compile(r'curl[^"]*"([^"]*bv-brc\.org[^"]*)"', re.IGNORECASE)
                curl_matches = curl_pattern.findall(action_text)
                commands.extend([f'curl "{match}"' for match in curl_matches])
                
                # Also try without quotes
                if not commands:
                    curl_pattern2 = re.compile(r'curl\s+([^\s]*bv-brc\.org[^\s]*)', re.IGNORECASE)
                    curl_matches2 = curl_pattern2.findall(action_text)
                    commands.extend([f'curl {match}' for match in curl_matches2])
                    
        except json.JSONDecodeError:
            # Fallback to regex extraction
            curl_pattern = re.compile(r'curl[^"]*"([^"]*bv-brc\.org[^"]*)"', re.IGNORECASE)
            curl_matches = curl_pattern.findall(action_text)
            commands.extend([f'curl "{match}"' for match in curl_matches])
            
        return commands
        
    def execute_bvbrc_commands(self, commands: List[str]) -> List[Dict]:
        """Execute a list of BV-BRC commands and return results"""
        results = []
        
        for command in commands:
            stdout, stderr = self.bvbrc_executor.run_query(command)
            is_success, reason = self.classifier.classify_execution(stdout, stderr)
            
            result = {
                'command': command,
                'stdout': stdout,
                'stderr': stderr,
                'is_success': is_success,
                'reason': reason
            }
            results.append(result)
            
        return results
        
    def process_response(self, response_text: str, user_query: str, 
                        max_iterations: int = 3) -> str:
        """
        Process a response with pseudo-chain editing.
        
        Args:
            response_text: The generated response text
            user_query: The original user query
            max_iterations: Maximum number of fix iterations per action
            
        Returns:
            Updated response text with execution results
        """
        steps = self.parse_reasoning_steps(response_text)
        
        # Process each action step
        for step_index, step in enumerate(steps):
            if step.step_type != StepType.ACTION:
                continue
                
            # Extract BV-BRC commands from this action
            commands = self.extract_bvbrc_commands(step.text)
            if not commands:
                continue
                
            # Execute commands with retry logic
            for iteration in range(max_iterations):
                execution_results = self.execute_bvbrc_commands(commands)
                
                # Check if any command failed
                failed_results = [r for r in execution_results if not r['is_success']]
                
                if not failed_results:
                    # All commands succeeded
                    step.execution_result = {
                        'commands': commands,
                        'results': execution_results,
                        'success': True
                    }
                    
                    # Check if we should update subsequent steps
                    if self.step_updater.should_update_subsequent_steps(
                        execution_results[0], step_index, len(steps)):
                        steps = self.step_updater.update_subsequent_steps(
                            steps, step_index, execution_results[0], user_query)
                    
                    break
                else:
                    # Try to fix failed commands
                    if iteration < max_iterations - 1:  # Don't fix on last iteration
                        for failed_result in failed_results:
                            fixed_command = self.action_fixer.fix_failed_action(
                                failed_result['command'], failed_result, user_query)
                            # Replace the failed command
                            cmd_index = commands.index(failed_result['command'])
                            commands[cmd_index] = fixed_command
                    else:
                        # Max iterations reached, mark as failed
                        step.execution_result = {
                            'commands': commands,
                            'results': execution_results,
                            'success': False
                        }
        
        # Reconstruct the response with execution results
        return self.reconstruct_response(steps)
        
    def reconstruct_response(self, steps: List[ReasoningStep]) -> str:
        """
        Reconstruct the response text from processed steps.
        Includes execution results where available.
        """
        reconstructed_parts = []
        
        for step in steps:
            if step.step_type == StepType.ACTION and step.execution_result:
                # Include the original action
                reconstructed_parts.append(f"<action>\n{step.text}\n</action>")
                
                # Add execution results
                results = step.execution_result['results']
                if results:
                    # Add a summary of execution results
                    reconstructed_parts.append("\n<execution_results>")
                    for result in results:
                        if result['is_success']:
                            # Truncate long outputs
                            stdout = result['stdout']
                            if len(stdout) > 500:
                                stdout = stdout[:500] + "... (truncated)"
                            reconstructed_parts.append(f"Command: {result['command']}")
                            reconstructed_parts.append(f"Result: {stdout}")
                        else:
                            reconstructed_parts.append(f"Command: {result['command']}")
                            reconstructed_parts.append(f"Error: {result['reason']}")
                    reconstructed_parts.append("</execution_results>\n")
            else:
                # Include the step as-is
                reconstructed_parts.append(step.text)
        
        return "\n".join(reconstructed_parts)


# Factory function for easy instantiation
def create_execution_only_processor(bvbrc_timeout: int = 30) -> ExecutionOnlyProcessor:
    """Create an ExecutionOnlyProcessor with default components"""
    bvbrc_executor = BVBRCShellExecutor(timeout=bvbrc_timeout)
    action_fixer = BVBRCActionFixer()
    step_updater = SubsequentStepUpdater()
    
    return ExecutionOnlyProcessor(
        bvbrc_executor=bvbrc_executor,
        action_fixer=action_fixer,
        step_updater=step_updater
    ) 