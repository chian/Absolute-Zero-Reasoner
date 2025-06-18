#!/usr/bin/env python3

import pandas as pd
import json

# Read the bio parquet files to see the structure  
print("=== TRAIN.PARQUET ===")
train_df = pd.read_parquet('data/bio/train.parquet')
print('Columns:', train_df.columns.tolist())
print('Shape:', train_df.shape)
print('\nFirst row:')
for col in train_df.columns:
    value = train_df[col].iloc[0]
    print(f'{col}: {type(value)} = {value}')
    
print('\nSample data:')
print(train_df.head(2).to_dict('records'))

print("\n" + "="*50)
print("=== TEST.PARQUET ===")
test_df = pd.read_parquet('data/bio/test.parquet')
print('Columns:', test_df.columns.tolist())
print('Shape:', test_df.shape)
print('\nFirst row:')
for col in test_df.columns:
    value = test_df[col].iloc[0]
    print(f'{col}: {type(value)} = {value}') 