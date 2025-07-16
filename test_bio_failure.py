#!/usr/bin/env python3
"""
Test script to manually run bio reasoning pipeline and show failures.
"""

import os
import sys
sys.path.append('.')

from absolute_zero_reasoner.data_construction.constructor import get_gen_bio_bvbrc_prompt
from absolute_zero_reasoner.utils.bvbrc_action_processor import BVBRCActionProcessor
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import SamplingParams
import torch

def test_bio_reasoning_failure():
    """Test a single bio reasoning query to see the failure."""
    
    # Test query
    user_query = "How many PATRIC CDS (Coding DNA Sequences) does Halorubrum xinjiangense strain CGMCC 1.3527 have?"
    
    print("=" * 80)
    print("TESTING BIO REASONING PIPELINE")
    print("=" * 80)
    print(f"User Query: {user_query}")
    print()
    
    # Step 1: Generate superprompt
    try:
        print("Step 1: Generating superprompt...")
        prompt_data = get_gen_bio_bvbrc_prompt(user_query)
        prompt = prompt_data['prompt'][0]['content']
        print("✅ Superprompt generated successfully!")
        print(f"Prompt length: {len(prompt)} characters")
        print()
        print("Generated prompt:")
        print("-" * 40)
        print(prompt)  # Show full prompt without truncation
        print("-" * 40)
        print()
    except Exception as e:
        print(f"❌ Superprompt generation failed: {e}")
        return
    
    # Step 2: Get LLM response (using a small model for testing)
    try:
        print("Step 2: Getting LLM response...")
        
        # Load the same model used in training
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"  # Testing larger Llama-based model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
        
        # Add padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Generate response using chat format
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
        
        # Move inputs to the same device as model
        device = next(model.parameters()).device
        inputs = inputs.to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=2048,  # Increased for complete responses
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the input prompt from response
        llm_response = response[len(tokenizer.decode(inputs[0], skip_special_tokens=True)):]
        
        print("✅ LLM response generated!")
        print("LLM Response:")
        print("-" * 40)
        print(llm_response)
        print("-" * 40)
        print()
        
    except Exception as e:
        print(f"❌ LLM response generation failed: {e}")
        # Use a mock response for testing execution
        llm_response = """
<think>
I need to find information about Halorubrum xinjiangense strain CGMCC 1.3527 and count its CDS features.
</think>
<action>
[
    "curl -s 'https://www.bv-brc.org/api-bulk/genome/?and(eq(genus,Halorubrum),eq(species,xinjiangense))&select(genome_id,genome_name)'",
    "curl -s 'https://www.bv-brc.org/api-bulk/feature/?and(eq(genome_id,1234567.3),eq(feature_type,CDS))&select(patric_id)&limit(1)'"
]
</action>
"""
        print("Using mock LLM response for testing execution:")
        print("-" * 40)
        print(llm_response)
        print("-" * 40)
        print()
    
    # Step 3: Execute and show failure
    try:
        print("Step 3: Executing commands...")
        
        # Create a mock actor rollout for testing
        class MockInferenceEngine:
            def __init__(self, model, tokenizer):
                self.model = model
                self.tokenizer = tokenizer
                
            def generate(self, prompts, sampling_params, prompt_token_ids, use_tqdm=False):
                try:
                    # Use the actual model to generate
                    inputs = torch.tensor(prompt_token_ids[0]).unsqueeze(0).to(self.model.device)
                    with torch.no_grad():
                        outputs = self.model.generate(
                            inputs,
                            max_new_tokens=sampling_params.max_tokens,
                            temperature=sampling_params.temperature,
                            pad_token_id=self.tokenizer.eos_token_id,
                            do_sample=sampling_params.temperature > 0
                        )
                    response_tokens = outputs[0][len(inputs[0]):]  # Remove prompt tokens
                    return [[response_tokens.cpu()], None]  # Return format expected by processor
                except Exception as e:
                    print(f"Mock inference engine error: {e}")
                    # Return a simple mock response token sequence for testing
                    mock_response = "FAILURE: API error - command syntax invalid"
                    mock_tokens = torch.tensor(self.tokenizer.encode(mock_response))
                    return [[mock_tokens], None]

        class MockActorRolloutWG:
            def __init__(self, model, tokenizer):
                self.inference_engine = MockInferenceEngine(model, tokenizer)

        mock_actor_rollout_wg = MockActorRolloutWG(model, tokenizer)
        
        # Create the new BVBRCActionProcessor
        processor = BVBRCActionProcessor(
            actor_rollout_wg=mock_actor_rollout_wg,
            tokenizer=tokenizer,
            bvbrc_timeout=10,
            max_retries=3
        )
        print("✅ BVBRCActionProcessor created successfully")
        
        # Process the response
        processed = processor.process_response(llm_response, user_query)
        
        print("✅ Execution completed!")
        print("Execution Results:")
        print("-" * 40)
        print(f"Result type: {type(processed)}")
        print(f"Processed response length: {len(processed)} characters")
        print("-" * 40)
        print()
        
        print("FULL PROCESSED RESPONSE:")
        print("=" * 60)
        print(processed)
        print("=" * 60)
        print()
            
        # Mimic the current system's execution error handling
        success_count = 0
        error_count = 0
        execution_details = []
        
        if '<execution_results>' in processed:
            start = processed.find('<execution_results>')
            end = processed.find('</execution_results>') + len('</execution_results>')
            execution_block = processed[start:end]
            
            # Parse execution results exactly like the current system
            lines = execution_block.split('\n')
            current_command = None
            
            for line in lines:
                line = line.strip()
                if line.startswith('Command:'):
                    current_command = line[8:].strip()  # Remove "Command: "
                elif line.startswith('Result:'):
                    success_count += 1
                    result_data = line[7:].strip()  # Remove "Result: "
                    execution_details.append({
                        'command': current_command,
                        'is_success': True,
                        'output': result_data,
                        'reason': 'Command executed successfully'
                    })
                elif line.startswith('Error:'):
                    error_count += 1
                    error_reason = line[6:].strip()  # Remove "Error: "
                    execution_details.append({
                        'command': current_command,
                        'is_success': False,
                        'output': '',
                        'reason': error_reason
                    })
            
            print("DETAILED EXECUTION RESULTS:")
            print("=" * 60)
            print(execution_block)
            print("=" * 60)
            print()
            
            # Print execution analysis matching the training system
            print("EXECUTION ANALYSIS (Mimicking Training System):")
            print("=" * 50)
            for i, detail in enumerate(execution_details, 1):
                if detail['is_success']:
                    print(f"Command {i}: ✅ SUCCESS")
                    print(f"  Command: {detail['command']}")
                    print(f"  Reason: {detail['reason']}")
                    if detail['output']:
                        output_preview = detail['output'][:100] + "..." if len(detail['output']) > 100 else detail['output']
                        print(f"  Output: {output_preview}")
                else:
                    print(f"Command {i}: ❌ FAILED")
                    print(f"  Command: {detail['command']}")
                    print(f"  Reason: {detail['reason']}")
                print()
            
            # Print summary matching training metrics
            print("TRAINING METRICS SIMULATION:")
            print("-" * 30)
            print(f"has_actions: 1 (action block found)")
            print(f"actions_executed: {success_count + error_count}")
            print(f"successful_executions: {success_count}")
            print(f"failed_executions: {error_count}")
            if success_count + error_count > 0:
                success_rate = success_count / (success_count + error_count)
                print(f"execution_success_rate: {success_rate:.3f}")
            else:
                print(f"execution_success_rate: 0.000")
            print(f"has_execution_results: 1")
            print()
            
            if success_count > 0:
                print("🎉 TRAINING OUTCOME: POSITIVE REWARD - At least one successful execution!")
            else:
                print("💥 TRAINING OUTCOME: NEGATIVE REWARD - No successful executions")
            print("=" * 50)
        else:
            print("❌ No execution results found in response")
            print("TRAINING METRICS SIMULATION:")
            print("-" * 30)
            print(f"has_actions: {'1' if '<action>' in processed else '0'}")
            print(f"actions_executed: 0")
            print(f"successful_executions: 0")
            print(f"failed_executions: 0")
            print(f"execution_success_rate: 0.000")
            print(f"has_execution_results: 0")
            print("💥 TRAINING OUTCOME: NEGATIVE REWARD - No execution results")
        
    except Exception as e:
        print(f"❌ Execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_bio_reasoning_failure() 