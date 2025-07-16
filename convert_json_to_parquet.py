import pandas as pd
import json
import os

def convert_json_to_parquet(json_path, parquet_path):
    """
    Reads data from a JSON file, converts it to the format expected by the training pipeline,
    and saves it as a Parquet file.
    """
    # Ensure the output directory exists
    output_dir = os.path.dirname(parquet_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Read the JSON file
    print(f"Reading JSON file from: {json_path}")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        print(f"Successfully loaded {len(data)} records from {json_path}")
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return

    # Transform data to the format expected by the training pipeline
    transformed_data = []
    for item in data:
        if 'question' in item:
            # Add instructional text to the question
            instructional_text = "\n\nPlease provide your reasoning within <think></think> tags and the final answer within a \\boxed{} block. The value inside the \\boxed{} block should only be the number of the correct option."
            question_with_instruction = item['question'] + instructional_text

            # Create the format expected by the training pipeline
            transformed_item = {
                "prompt": [{"role": "user", "content": question_with_instruction}],
                "data_source": "gen_bio_llm",
                "problem": item['question'],
                "ability": "bio_llm",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": item.get('answer', ''),
                },
                "extra_info": {
                    "question": item['question'],
                    "answer": item.get('answer', ''),
                    **{k: v for k, v in item.items() if k not in ['question', 'answer'] and k in ['curriculum_order', 'metric', 'verification_mode', 'split']}
                }
            }
            transformed_data.append(transformed_item)

    # Create DataFrame and save to Parquet
    try:
        df = pd.DataFrame(transformed_data)
        df.to_parquet(parquet_path, index=False)
        print(f"Successfully converted {len(transformed_data)} records and saved to {parquet_path}")
        print(f"Columns in output file: {df.columns.tolist()}")
    except Exception as e:
        print(f"Error saving to Parquet file: {e}")


if __name__ == "__main__":
    # Define the input and output file paths
    json_input_path = "data/bio_questions.json"
    parquet_output_path = "data/bio_llm/bio_llm_questions.parquet"

    # Run the conversion
    convert_json_to_parquet(json_input_path, parquet_output_path) 