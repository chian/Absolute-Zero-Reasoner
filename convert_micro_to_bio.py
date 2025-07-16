#!/usr/bin/env python3
"""
Script to convert micro-mc-gpt41-clean.json format to bio_bvbrc_questions.json format
"""

import json
import re
import sys

def extract_answer_number(answer_text):
    """Extract just the number from answer text like 'The correct answer is 3) Production of...'"""
    # Look for patterns like "The correct answer is 3)" or "3)" or just "3"
    patterns = [
        r'The correct answer is (\d+)',
        r'(\d+)\)',
        r'^(\d+)$'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, answer_text)
        if match:
            return match.group(1)
    
    # If no pattern matches, return the original text
    return answer_text

def convert_micro_to_bio(input_file, output_file):
    """Convert micro-mc-gpt41-clean.json to bio_bvbrc_questions.json format"""
    
    # Read input file
    with open(input_file, 'r', encoding='utf-8') as f:
        micro_data = json.load(f)
    
    # Convert to bio format
    bio_data = []
    
    for item in micro_data:
        # Extract just the number from the answer
        answer_number = extract_answer_number(item['answer'])
        
        # Create bio format entry
        bio_entry = {
            "question": item['question'],
            "answer": f"\\boxed{{{answer_number}}}",
            "verification_mode": "exact",
            "curriculum_order": 1
        }
        
        bio_data.append(bio_entry)
    
    # Write output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(bio_data, f, indent=2, ensure_ascii=False)
    
    print(f"Converted {len(bio_data)} questions from {input_file} to {output_file}")

def main():
    input_file = "data/micro-mc-gpt41-clean.json"
    output_file = "data/micro_converted_to_bio.json"
    
    try:
        convert_micro_to_bio(input_file, output_file)
    except FileNotFoundError:
        print(f"Error: Could not find input file {input_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 