"""
BV-BRC Action Processor for executing commands without Ray calls.
Separate component that sits between generation and reward evaluation.
"""

import json
import re
import os
import openai
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from absolute_zero_reasoner.utils.code_utils.bvbrc_executor import BVBRCShellExecutor
from absolute_zero_reasoner.utils.logging_utils.stdout import PrettyPrinter


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


def query_llm(client, prompt, config):
    """
    Make an LLM API call with the given client, prompt, and configuration.
    
    Args:
        client: The LLM client instance (Together or OpenAI)
        prompt: The prompt string to send
        config: Dictionary containing model configuration
        
    Returns:
        str: The response text from the LLM
    """
    try:
        # Create messages list with the prompt
        messages = []
        for message_template in config["messages"]:
            if "{user_query}" in message_template["content"]:
                # Replace template placeholder with actual prompt
                content = message_template["content"].format(user_query=prompt)
            else:
                content = message_template["content"]
            
            messages.append({
                "role": message_template["role"],
                "content": content
            })
        
        # Make the API call
        response = client.chat.completions.create(
            model=config["model_id"],
            messages=messages,
            max_tokens=config.get("max_tokens", 1000),
            temperature=config.get("temperature", 0.7)
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error calling LLM: {str(e)}"


# OpenAI GPT-4o-mini configuration
COMMAND_FIXING_CONFIG = {
    "api_key": os.getenv('OPENAI_API_KEY'),
    "endpoint": "https://api.openai.com/v1",
    "model_id": "gpt-4o-mini",
    "client": openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY')) if os.getenv('OPENAI_API_KEY') else None,
    "messages": [
        {"role": "user", "content": "You are a helpful assistant.\n\n{user_query}"}
    ],
    "max_tokens": 1000,
    "temperature": 0.7
}

class BVBRCActionProcessor:
    """
    Processes responses with BV-BRC actions and executes commands.
    NO RAY CALLS - completely separate from reward evaluation.
    """
    
    def __init__(self, tokenizer=None, bvbrc_timeout: int = 30, max_retries: int = 3):
        """
        Initialize the BV-BRC Action Processor.
        
        Args:
            tokenizer: Not used anymore (kept for compatibility) 
            bvbrc_timeout: Timeout for BV-BRC command execution
            max_retries: Maximum number of retries for failed commands
        """
        self.bvbrc_executor = BVBRCShellExecutor(timeout=bvbrc_timeout)
        self.max_retries = max_retries
        
        # Initialize OpenAI client for command fixing
        self.llm_client = COMMAND_FIXING_CONFIG["client"]
        self.llm_config = COMMAND_FIXING_CONFIG
    
    def fix_failed_command(self, failed_command: str, error_output: str) -> Optional[str]:
        """
        Use LLM to fix a failed BV-BRC command.
        
        Args:
            failed_command: The command that failed
            error_output: The error message from the failed command
            
        Returns:
            Fixed command string or None if fixing failed
        """
        if not self.llm_client:
            return None
            
        prompt = f"""Fix this BV-BRC command that failed:

Command: {failed_command}
Error: {error_output}

Return only the corrected command:"""

        try:
            fixed_command = query_llm(self.llm_client, prompt, self.llm_config)
            return fixed_command.strip() if fixed_command else None
        except Exception:
            return None
    
    def extract_bvbrc_commands(self, action_text: str) -> List[Dict[str, str]]:
        """
        Extract BV-BRC curl commands from action text.
        
        Args:
            action_text: Text containing action commands
            
        Returns:
            List of command dictionaries
        """
        commands = []
        
        try:
            # Try to parse as JSON first
            if action_text.strip().startswith('['):
                action_data = json.loads(action_text)
                for item in action_data:
                    if isinstance(item, dict) and 'action_input' in item:
                        commands.append({
                            "action": item.get("action", "bash"),
                            "action_input": item['action_input']
                        })
            elif action_text.strip().startswith('{'):
                action_data = json.loads(action_text)
                if 'action_input' in action_data:
                    commands.append({
                        "action": action_data.get("action", "bash"),
                        "action_input": action_data['action_input']
                    })
            else:
                # Extract curl commands directly using regex
                curl_pattern = re.compile(r'curl[^"]*"([^"]*bv-brc\.org[^"]*)"', re.IGNORECASE)
                curl_matches = curl_pattern.findall(action_text)
                commands.extend([{
                    "action": "bash",
                    "action_input": f'curl "{match}"'
                } for match in curl_matches])
                
        except json.JSONDecodeError:
            # Fallback to regex extraction
            curl_pattern = re.compile(r'curl[^"]*"([^"]*bv-brc\.org[^"]*)"', re.IGNORECASE)
            curl_matches = curl_pattern.findall(action_text)
            commands.extend([{
                "action": "bash", 
                "action_input": f'curl "{match}"'
            } for match in curl_matches])
            
        return commands
    
    def classify_execution_result(self, stdout: str, stderr: str, user_query: str, command: str) -> Tuple[bool, str]:
        """
        Classify execution result as success or failure using simple rules.
        NO LLM CALLS.
        
        Args:
            stdout: Standard output from command
            stderr: Standard error from command  
            user_query: Original user query (unused)
            command: The executed command (unused)
            
        Returns:
            Tuple of (is_success, classification_text)
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
        if "not found" in stdout.lower():
            return False, "Resource not found"
        if "invalid" in stdout.lower():
            return False, "Invalid request"
            
        # Simple success detection - if we got JSON-like output
        if stdout.strip().startswith('{') or stdout.strip().startswith('['):
            return True, "Valid JSON response received"
        
        # If we have substantial output without error indicators
        if len(stdout.strip()) > 10:
            return True, "Command executed successfully"
            
        return False, "Unclear result"
    
    def process_response(self, response_text: str, user_query: str) -> str:
        """
        Main processing function: parse actions, execute commands.
        NO RAY CALLS OR LLM RETRIES.
        
        Args:
            response_text: Generated response containing <action> tags
            user_query: Original user query
            
        Returns:
            Processed response with <execution_results> tags
        """
        # Parse for action blocks
        action_pattern = re.compile(r'<action>\s*(.*?)\s*</action>', re.DOTALL | re.IGNORECASE)
        action_matches = list(action_pattern.finditer(response_text))
        
        if not action_matches:
            PrettyPrinter.status("BVBRC_PROC", "No action tags found, returning original response", "warning")
            return response_text
            
        # Track execution results for each action
        processed_parts = []
        current_pos = 0
        
        for action_match in action_matches:
            # Add text before this action
            if action_match.start() > current_pos:
                processed_parts.append(response_text[current_pos:action_match.start()])
            
            # Add the action
            action_content = action_match.group(1).strip()
            processed_parts.append(f"<action>\n{action_content}\n</action>")
            
            # Extract and execute commands
            commands = self.extract_bvbrc_commands(action_content)
            
            if commands:
                execution_results = []
                
                for cmd in commands:
                    PrettyPrinter.status("BVBRC_EXEC", f"Executing: {cmd['action_input']}", "info")
                    
                    # Execute with retries and LLM fixing
                    success = False
                    current_command = cmd['action_input']
                    
                    for retry in range(self.max_retries + 1):
                        stdout, stderr = self.bvbrc_executor.run_query(current_command)
                        is_success, classification = self.classify_execution_result(
                            stdout, stderr, user_query, current_command
                        )
                        
                        if is_success:
                            execution_results.append({
                                "command": current_command,
                                "result": stdout,
                                "status": "success"
                            })
                            PrettyPrinter.status("BVBRC_EXEC", "Command succeeded", "success")
                            success = True
                            break
                        else:
                            # Try to fix with LLM if not last retry
                            if retry < self.max_retries:
                                fixed_command = self.fix_failed_command(current_command, stderr if stderr else stdout)
                                if fixed_command and fixed_command != current_command:
                                    PrettyPrinter.status("BVBRC_EXEC", f"Trying LLM fix: {fixed_command}", "info")
                                    current_command = fixed_command
                                    continue
                    
                    if not success:
                        execution_results.append({
                            "command": current_command, 
                            "result": stderr if stderr else stdout,
                            "status": "failed"
                        })
                        PrettyPrinter.status("BVBRC_EXEC", f"Command failed: {classification}", "warning")
                
                # Add execution results
                execution_results_text = "\n<execution_results>\n"
                for result in execution_results:
                    if result["status"] == "success":
                        execution_results_text += f"Command: {result['command']}\n"
                        execution_results_text += f"Result: {result['result']}\n"
                    else:
                        execution_results_text += f"Command: {result['command']}\n"
                        execution_results_text += f"Error: {result['result']}\n"
                execution_results_text += "</execution_results>\n"
                
                processed_parts.append(execution_results_text)
            
            current_pos = action_match.end()
        
        # Add any remaining text
        if current_pos < len(response_text):
            processed_parts.append(response_text[current_pos:])
            
        return "".join(processed_parts) 