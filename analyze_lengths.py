#!/usr/bin/env python3

import re
import statistics

def analyze_question_lengths():
    """
    Analyze the length of questions to see if corrupted ones are shorter.
    """
    
    # Expected instruction text from convert_json_to_parquet.py
    EXPECTED_INSTRUCTION = "Please provide your reasoning within <think></think> tags and the final answer within a \\boxed{} block. The value inside the \\boxed{} block should only be the number of the correct option."
    
    # Read extracted questions
    with open('extracted_questions.txt', 'r') as f:
        content = f.read()
    
    # Split by question markers
    question_blocks = re.split(r'--- Question \d+ ---', content)
    
    print("QUESTION LENGTH ANALYSIS")
    print("=" * 80)
    print()
    
    categories = {
        'clean': [],
        'response_contamination': [],
        'truncated_instructions': [],
        'instruction_variations': []
    }
    
    for i, block in enumerate(question_blocks[1:], 1):  # Skip first empty block
        block = block.strip()
        if block.endswith('---------------------'):
            block = block[:-21].strip()
        
        question_length = len(block)
        
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
            categories['response_contamination'].append({
                'question_num': i,
                'length': question_length,
                'content': block[:100] + "..." if len(block) > 100 else block
            })
            continue
        
        # Check for instruction text issues
        if EXPECTED_INSTRUCTION in block:
            # Perfect match
            categories['clean'].append({
                'question_num': i,
                'length': question_length,
                'content': block[:100] + "..." if len(block) > 100 else block
            })
        elif "Please provide your reasoning within <think></think> tags" in block:
            # Has the start but check if it's complete/correct
            instruction_start = block.find("Please provide your reasoning within <think></think> tags")
            instruction_part = block[instruction_start:]
            
            if instruction_part != EXPECTED_INSTRUCTION:
                if len(instruction_part) < len(EXPECTED_INSTRUCTION):
                    # Truncated
                    categories['truncated_instructions'].append({
                        'question_num': i,
                        'length': question_length,
                        'content': block[:100] + "..." if len(block) > 100 else block
                    })
                else:
                    # Variation
                    categories['instruction_variations'].append({
                        'question_num': i,
                        'length': question_length,
                        'content': block[:100] + "..." if len(block) > 100 else block
                    })
            else:
                categories['clean'].append({
                    'question_num': i,
                    'length': question_length,
                    'content': block[:100] + "..." if len(block) > 100 else block
                })
        else:
            # No instruction text found at all
            categories['truncated_instructions'].append({
                'question_num': i,
                'length': question_length,
                'content': block[:100] + "..." if len(block) > 100 else block
            })
    
    # Calculate statistics for each category
    print("LENGTH STATISTICS BY CATEGORY:")
    print("-" * 50)
    
    for category, questions in categories.items():
        if not questions:
            continue
            
        lengths = [q['length'] for q in questions]
        
        print(f"\n{category.upper().replace('_', ' ')} ({len(questions)} questions):")
        print(f"  Min length: {min(lengths)}")
        print(f"  Max length: {max(lengths)}")
        print(f"  Mean length: {statistics.mean(lengths):.1f}")
        print(f"  Median length: {statistics.median(lengths):.1f}")
        
        # Show some examples
        print(f"  Examples:")
        sorted_by_length = sorted(questions, key=lambda x: x['length'])
        
        # Show shortest
        shortest = sorted_by_length[0]
        print(f"    Shortest (Q{shortest['question_num']}, {shortest['length']} chars): {shortest['content']}")
        
        # Show longest
        longest = sorted_by_length[-1]
        print(f"    Longest (Q{longest['question_num']}, {longest['length']} chars): {longest['content']}")
        
        # Show median
        if len(sorted_by_length) > 2:
            median_idx = len(sorted_by_length) // 2
            median_q = sorted_by_length[median_idx]
            print(f"    Median (Q{median_q['question_num']}, {median_q['length']} chars): {median_q['content']}")
    
    # Compare categories
    print("\n" + "=" * 80)
    print("COMPARISON:")
    print("-" * 50)
    
    if categories['clean'] and categories['response_contamination']:
        clean_lengths = [q['length'] for q in categories['clean']]
        contaminated_lengths = [q['length'] for q in categories['response_contamination']]
        
        print(f"Clean questions average length: {statistics.mean(clean_lengths):.1f}")
        print(f"Response contaminated average length: {statistics.mean(contaminated_lengths):.1f}")
        print(f"Difference: {statistics.mean(contaminated_lengths) - statistics.mean(clean_lengths):+.1f}")
    
    if categories['clean'] and categories['truncated_instructions']:
        clean_lengths = [q['length'] for q in categories['clean']]
        truncated_lengths = [q['length'] for q in categories['truncated_instructions']]
        
        print(f"Clean questions average length: {statistics.mean(clean_lengths):.1f}")
        print(f"Truncated instructions average length: {statistics.mean(truncated_lengths):.1f}")
        print(f"Difference: {statistics.mean(truncated_lengths) - statistics.mean(clean_lengths):+.1f}")
    
    # Look for patterns in truncated questions
    if categories['truncated_instructions']:
        print(f"\nTRUNCATED QUESTIONS ANALYSIS:")
        print("-" * 50)
        
        truncated = categories['truncated_instructions']
        very_short = [q for q in truncated if q['length'] < 500]
        medium = [q for q in truncated if 500 <= q['length'] < 1000]
        long = [q for q in truncated if q['length'] >= 1000]
        
        print(f"Very short (<500 chars): {len(very_short)}")
        print(f"Medium (500-1000 chars): {len(medium)}")
        print(f"Long (>=1000 chars): {len(long)}")
        
        if very_short:
            print(f"\nSample very short truncated question:")
            sample = very_short[0]
            print(f"Q{sample['question_num']} ({sample['length']} chars):")
            print(f"'{sample['content']}'")

if __name__ == "__main__":
    analyze_question_lengths() 