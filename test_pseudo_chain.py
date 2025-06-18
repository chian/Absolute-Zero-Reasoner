#!/usr/bin/env python3
"""
Test script for the pseudo-chain processor implementation.
"""

import sys
import os

# Add the absolute_zero_reasoner to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from absolute_zero_reasoner.utils.pseudo_chain_processor import create_execution_only_processor


def test_basic_processing():
    """Test basic pseudo-chain processing functionality"""
    print("Testing basic pseudo-chain processing...")
    
    # Create processor
    processor = create_execution_only_processor(bvbrc_timeout=10)
    
    # Test response with action
    test_response = """
I need to find Pseudomonas aeruginosa genomes. Let me query the BV-BRC database.

<action>
{
    "action": "bash",
    "action_input": "curl -s \"https://www.bv-brc.org/api-bulk/genome/?and(eq(genus,Pseudomonas),eq(species,aeruginosa))&select(genome_id)&limit(3)\""
}
</action>

Based on the results, I can analyze the genome data.
"""
    
    user_query = "List all Pseudomonas aeruginosa genome IDs."
    
    try:
        processed_response = processor.process_response(test_response, user_query)
        print("✅ Processing completed successfully!")
        print(f"Original length: {len(test_response)}")
        print(f"Processed length: {len(processed_response)}")
        print("\n--- Processed Response ---")
        print(processed_response[:500] + "..." if len(processed_response) > 500 else processed_response)
        
        # Check if execution results were added
        if "<execution_results>" in processed_response:
            print("✅ Execution results were added to the response")
        else:
            print("⚠️ No execution results found in processed response")
            
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()


def test_bio_reward_manager():
    """Test the bio reasoning reward manager"""
    print("\nTesting bio reasoning reward manager...")
    
    try:
        from transformers import AutoTokenizer
        from absolute_zero_reasoner.rewards.bio_bvbrc_reward_manager import BioReasoningRewardManager
        
        # Use a simple tokenizer for testing
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        reward_manager = BioReasoningRewardManager(
            tokenizer=tokenizer,
            num_examine=1,
            split='test',
            reward_fn_extraction_type='none',
            splitter='Human:',
            output_path='/tmp',
            debug=True,
            enable_pseudo_chain=True,
        )
        
        print("✅ Bio reasoning reward manager created successfully!")
        
    except Exception as e:
        print(f"❌ Bio reward manager creation failed: {e}")
        import traceback
        traceback.print_exc()


def test_step_parsing():
    """Test the step parsing functionality"""
    print("\nTesting step parsing...")
    
    processor = create_execution_only_processor(bvbrc_timeout=5)
    
    test_response = """
First, I need to understand what genomes are available.

<action>
{
    "action": "bash", 
    "action_input": "curl \"https://www.bv-brc.org/api-bulk/genome/?limit(1)\""
}
</action>

Now let me search for specific organisms.

<action>
{
    "action": "bash",
    "action_input": "curl \"https://www.bv-brc.org/api-bulk/genome/?eq(genus,Escherichia)&limit(2)\""
}
</action>

The answer is that we found the genomes successfully.
"""
    
    steps = processor.parse_reasoning_steps(test_response)
    
    print(f"✅ Parsed {len(steps)} steps:")
    for i, step in enumerate(steps):
        print(f"  Step {i+1}: {step.step_type.value} ({len(step.text)} chars)")
        if step.step_type.value == "action":
            commands = processor.extract_bvbrc_commands(step.text)
            print(f"    Commands found: {len(commands)}")


if __name__ == "__main__":
    print("🧪 Testing Pseudo-Chain Processor Implementation")
    print("=" * 50)
    
    test_basic_processing()
    test_bio_reward_manager()
    test_step_parsing()
    
    print("\n✅ All tests completed!") 