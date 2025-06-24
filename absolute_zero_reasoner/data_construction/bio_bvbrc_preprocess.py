#!/usr/bin/env python3
"""
Preprocess the bio BV-BRC dataset to parquet format following the existing data pattern
"""

import json
import os
import argparse
from absolute_zero_reasoner.data_construction.constructor import get_gen_bio_bvbrc_prompt


def preprocess_bio_bvbrc_data(bio_data_path: str, local_dir: str):
    """Convert bio_questions.json to the standard parquet format with curriculum support"""
    
    # Load the JSON data
    with open(bio_data_path, 'r') as f:
        bio_data = json.load(f)
    
    # Sort by curriculum_order for proper progression
    bio_data = sorted(bio_data, key=lambda x: x.get('curriculum_order', 999))
    
    # Create training data following the GSM8K pattern
    train_data = []
    test_data = []
    
    for idx, item in enumerate(bio_data):
        # Generate the prompt using the existing function
        prompt_data = get_gen_bio_bvbrc_prompt(item['question'])
        
        # Create data entry following the standard format
        data_entry = {
            "data_source": "bio_bvbrc",
            "prompt": prompt_data["prompt"],
            "ability": "bio_bvbrc",
            "reward_model": {
                "style": "rule",
                "ground_truth": json.dumps(item['answer'])
            },
            "extra_info": {
                'split': 'train' if idx < len(bio_data) * 0.8 else 'test',
                'index': idx,
                'question': item['question'],
                'answer': json.dumps(item['answer']) if not isinstance(item['answer'], str) else item['answer'],
                'curriculum_order': item.get('curriculum_order', 999),
                'verification_mode': item.get('verification_mode', 'exact'),
                'raw_answer': item['answer']  # Keep original format for verification
            }
        }
        
        # Split into train/test (80/20)
        if idx < len(bio_data) * 0.8:
            train_data.append(data_entry)
        else:
            test_data.append(data_entry)
    
    # Create output directory
    os.makedirs(local_dir, exist_ok=True)
    
    # Convert to parquet using the existing pattern
    import pandas as pd
    
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)
    
    train_df.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_df.to_parquet(os.path.join(local_dir, 'test.parquet'))
    
    print(f"Created train.parquet with {len(train_data)} samples")
    print(f"Created test.parquet with {len(test_data)} samples")
    print("Sample entry structure:")
    print(train_df.iloc[0].to_dict())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bio_data_path', default='data/bio_questions.json', 
                       help='Path to bio_questions.json file')
    parser.add_argument('--local_dir', default='data/bio',
                       help='Output directory for parquet files')
    
    args = parser.parse_args()
    
    preprocess_bio_bvbrc_data(args.bio_data_path, args.local_dir) 