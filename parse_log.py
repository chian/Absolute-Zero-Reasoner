import re
import sys

def parse_log_file(log_file_path):
    """
    Parses a log file to extract the full text of questions, stripping out ANSI color codes.
    """
    try:
        with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: Log file not found at {log_file_path}")
        return

    # Regex to remove ANSI escape codes
    ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    clean_content = ansi_escape.sub('', content)

    # Use a regular expression to find all text between [QUESTION] and [RESPONSE]
    # The re.DOTALL flag allows '.' to match newlines
    question_blocks = re.findall(r"ℹ \[QUESTION\](.*?)ℹ \[RESPONSE\]", clean_content, re.DOTALL)

    if not question_blocks:
        print("No questions found in the log file.")
        return

    for i, block in enumerate(question_blocks):
        # Clean up the extracted block by removing log prefixes
        cleaned_block = re.sub(r"\(main_task pid=\d+\)\s*", "", block).strip()
        
        print(f"--- Question {i + 1} ---")
        print(cleaned_block)
        print("-" * (20 + len(str(i + 1))))
        print("\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        parse_log_file(sys.argv[1])
    else:
        print("Usage: python parse_log.py <path_to_log_file>") 