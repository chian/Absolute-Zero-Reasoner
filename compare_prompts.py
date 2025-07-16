#!/usr/bin/env python3

import re

def analyze_prompt_errors():
    """
    Compare the expected prompt format with what's in extracted_questions.txt
    and identify all errors and deviations.
    """
    
    # Expected instruction text from convert_json_to_parquet.py
    EXPECTED_INSTRUCTION = "Please provide your reasoning within <think></think> tags and the final answer within a \\boxed{} block. The value inside the \\boxed{} block should only be the number of the correct option."
    
    # Read extracted questions
    with open('extracted_questions.txt', 'r') as f:
        content = f.read()
    
    # Split by question markers
    question_blocks = re.split(r'--- Question \d+ ---', content)
    
    print("PROMPT ERROR ANALYSIS")
    print("=" * 80)
    print(f"Expected instruction: {EXPECTED_INSTRUCTION}")
    print("=" * 80)
    print()
    
    total_questions = len(question_blocks) - 1  # First block is empty
    corrupted_count = 0
    truncated_count = 0
    variation_count = 0
    clean_count = 0
    
    errors = {
        'response_contamination': [],
        'truncated_instructions': [],
        'instruction_variations': [],
        'clean_questions': []
    }
    
    for i, block in enumerate(question_blocks[1:], 1):  # Skip first empty block
        block = block.strip()
        if block.endswith('---------------------'):
            block = block[:-21].strip()
        
        # Check for response contamination
        response_markers = [
            "Okay, so I'm trying to figure out",
            "Step-by-Step Explanation:",
            "Let me think about this",
            "Looking at this question",
            "I need to consider"
        ]
        
        has_response_contamination = any(marker in block for marker in response_markers)
        
        if has_response_contamination:
            corrupted_count += 1
            # Find where the contamination starts
            contamination_start = None
            for marker in response_markers:
                if marker in block:
                    contamination_start = block.find(marker)
                    break
            
            clean_part = block[:contamination_start] if contamination_start else block
            contaminated_part = block[contamination_start:] if contamination_start else ""
            
            errors['response_contamination'].append({
                'question_num': i,
                'clean_part': clean_part[:200] + "..." if len(clean_part) > 200 else clean_part,
                'contaminated_part': contaminated_part[:200] + "..." if len(contaminated_part) > 200 else contaminated_part
            })
            continue
        
        # Check for instruction text issues
        if EXPECTED_INSTRUCTION in block:
            # Perfect match
            clean_count += 1
            errors['clean_questions'].append(i)
        elif "Please provide your reasoning within <think></think> tags" in block:
            # Has the start but check if it's complete/correct
            instruction_start = block.find("Please provide your reasoning within <think></think> tags")
            instruction_part = block[instruction_start:]
            
            if instruction_part != EXPECTED_INSTRUCTION:
                if len(instruction_part) < len(EXPECTED_INSTRUCTION):
                    # Truncated
                    truncated_count += 1
                    errors['truncated_instructions'].append({
                        'question_num': i,
                        'found': instruction_part,
                        'expected': EXPECTED_INSTRUCTION
                    })
                else:
                    # Variation
                    variation_count += 1
                    errors['instruction_variations'].append({
                        'question_num': i,
                        'found': instruction_part,
                        'expected': EXPECTED_INSTRUCTION
                    })
            else:
                clean_count += 1
                errors['clean_questions'].append(i)
        else:
            # No instruction text found at all
            truncated_count += 1
            errors['truncated_instructions'].append({
                'question_num': i,
                'found': "NO INSTRUCTION TEXT FOUND",
                'expected': EXPECTED_INSTRUCTION
            })
    
    # Print summary
    print(f"SUMMARY:")
    print(f"Total questions analyzed: {total_questions}")
    print(f"Clean questions: {clean_count}")
    print(f"Response contamination: {corrupted_count}")
    print(f"Truncated instructions: {truncated_count}")
    print(f"Instruction variations: {variation_count}")
    print()
    
    # Print detailed errors
    if errors['response_contamination']:
        print("RESPONSE CONTAMINATION ERRORS:")
        print("-" * 40)
        for error in errors['response_contamination'][:5]:  # Show first 5
            print(f"Question {error['question_num']}:")
            print(f"  Clean part: {error['clean_part']}")
            print(f"  Contaminated part: {error['contaminated_part']}")
            print()
        if len(errors['response_contamination']) > 5:
            print(f"... and {len(errors['response_contamination']) - 5} more")
        print()
    
    if errors['truncated_instructions']:
        print("TRUNCATED INSTRUCTION ERRORS:")
        print("-" * 40)
        for error in errors['truncated_instructions'][:5]:  # Show first 5
            print(f"Question {error['question_num']}:")
            print(f"  Found: {error['found']}")
            print(f"  Expected: {error['expected']}")
            print()
        if len(errors['truncated_instructions']) > 5:
            print(f"... and {len(errors['truncated_instructions']) - 5} more")
        print()
    
    if errors['instruction_variations']:
        print("INSTRUCTION VARIATION ERRORS:")
        print("-" * 40)
        for error in errors['instruction_variations'][:5]:  # Show first 5
            print(f"Question {error['question_num']}:")
            print(f"  Found: {error['found']}")
            print(f"  Expected: {error['expected']}")
            print()
        if len(errors['instruction_variations']) > 5:
            print(f"... and {len(errors['instruction_variations']) - 5} more")
        print()
    
    print(f"Clean questions: {errors['clean_questions'][:10]}..." if len(errors['clean_questions']) > 10 else f"Clean questions: {errors['clean_questions']}")
    
    return errors

if __name__ == "__main__":
    analyze_prompt_errors() 