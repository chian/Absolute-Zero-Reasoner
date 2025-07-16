import re
import json
import ast
import os
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum

from absolute_zero_reasoner.utils.code_utils.bvbrc_executor import BVBRCShellExecutor


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


@dataclass
class AttemptResult:
    """Represents the result of a single attempt"""
    attempt_number: int
    response_text: str
    commands: List[str]
    execution_results: List[Dict]
    success: bool
    failure_reason: str = ""


class BioReasoningHelpers:
    """Helper functions for bio reasoning retry logic, adapted from solr-together.py"""
    
    @staticmethod
    def remove_think_tags(text: str) -> str:
        """Remove <think> tags and their contents from text."""
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    @staticmethod
    def query_llm(client, user_query, model_config):
        """Query an LLM with the given user query using the provided configuration."""
        model_id = model_config.get('model_id', model_config.get('model'))
        messages = model_config['messages']
        
        # Check if we're using a local model or OpenAI
        if hasattr(client, 'chat') and hasattr(client.chat, 'completions'):
            # OpenAI-style client
            params = {
                "model": model_id,
                "messages": messages
            }
            
            api_params = model_config.get('api_params', {})
            params.update(api_params)
            
            response = client.chat.completions.create(**params)
            return response.choices[0].message.content
        else:
            # Local model client - assume it's a simple callable
            # Extract just the user message for simpler local inference
            user_message = ""
            for msg in messages:
                if msg["role"] == "user":
                    user_message = msg["content"]
                    break
            
            if callable(client):
                result = client(user_message)
                return str(result) if result is not None else ""
            else:
                # Fallback: return empty response if client not properly configured
                return "Error: Local model client not properly configured"
    
    @staticmethod
    def classify_last_command_output_with_llm(user_query, cmd, observation_text, client):
        """Uses the LLM to classify the output of the last command."""
        observation_lines = observation_text.splitlines()[:100]
        observation_text = "\n".join(observation_lines)
        
        prompt = f"""
        You are evaluating if a command execution was TECHNICALLY SUCCESSFUL.

        User Query: "{user_query}"

        Here is the output of the command:
        ```
        {observation_text}
        ```

        Classify ONLY the TECHNICAL execution of this command:

        1. SUCCESS: Command executed correctly and returned valid, non-error data
           - No error messages, API errors, or undefined field errors
           - Returned properly formatted data
           - Output can be used for further analysis (even if incomplete)

        2. FAILURE: Command failed technically
           - Contains syntax errors
           - Shows API errors like "undefined field" or HTTP error codes
           - Returned error messages instead of data
           - Output cannot be used for further processing

        Important: This is ONLY about technical execution, NOT about answering the user's query.
        A command can succeed technically but not provide a complete answer.

        Classification: [SUCCESS/FAILURE]
        Brief explanation: 
        """
        
        model_config = {
            "model": getattr(client, 'default_model'),
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1000,
            "temperature": 0.7
        }
        
        return BioReasoningHelpers.query_llm(client, prompt, model_config)
    
    @staticmethod
    def derive_solution_with_llm(classification, client):
        """Uses the LLM to derive a solution based on the classification."""
        prompt = f"""
        Based on the following classification of a command output:
        
        Classification:
        {classification}
        
        Please provide a concise solution or answer based on this information.
        If the command was not successful, indicate what would be needed to answer the query.
        """
        
        model_config = {
            "model": getattr(client, 'default_model'),
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1000,
            "temperature": 0.7
        }
        
        return BioReasoningHelpers.query_llm(client, prompt, model_config)
    
    @staticmethod
    def update_commands_with_llm(user_query, commands, command_index, cmd, observation_text, classification, client):
        """Uses the LLM to update the command list based on the current command's output."""
        observation_lines = observation_text.splitlines()[:100]
        observation_text = "\n".join(observation_lines)
        
        prompt = f"""
        Based on the following command execution:
        
        User Query: "{user_query}"
        
        Command executed:
        {cmd}
        
        Command output:
        ```
        {observation_text}
        ```
        
        Classification:
        {classification}
        
        Please provide the next command to execute to answer the query.
        Return ONLY the command, nothing else.
        """
        
        model_config = {
            "model": getattr(client, 'default_model'),
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1000,
            "temperature": 0.7
        }
        
        next_cmd = BioReasoningHelpers.query_llm(client, prompt, model_config)
        new_cmd = {"action": commands[command_index]["action"], "action_input": next_cmd.strip()}
        commands[command_index + 1] = new_cmd
        return True, commands, "Commands updated based on execution result"
    
    @staticmethod
    def fix_failed_command_with_llm(user_query, cmd, observation_text, classification, client):
        """When a command fails, prompts the LLM to fix just that one command."""
        observation_lines = observation_text.splitlines()[:100]
        observation_text = "\n".join(observation_lines)
        
        # Get relevant API documentation based on the command and error
        api_context, doc_filename = BioReasoningHelpers.get_relevant_api_docs(cmd['action_input'], observation_text)
        
        fix_prompt = f"""
        User Query: {user_query}
        
        The following command failed:
        ```
        {cmd['action_input']}
        ```
        
        Output:
        {observation_text}
        
        Classification:
        {classification}
        
        {api_context}
        
        IMPORTANT: ONLY fix this specific command syntax. DO NOT:
        - Suggest multiple commands or a completely different approach
        - Recommend tools or utilities not mentioned in the original command
        - Propose alternative APIs or endpoints
        - Change the fundamental approach
        - DO NOT TRY TO USE ALTERNATIVES TO THE CORRECT BV-BRC SOLR CALL
        - USE BV-BRC ONLY - USE BV-BRC ONLY - USE BV-BRC ONLY
            --This means the "action" should be "bash" since that's how you make a SOLR query
            --This also means the "action_input" should start with "curl" since that's how you make a SOLR query
        - Consider if you should think about using a different endpoint.
            --For example: consider genome to find genome ids before using genome_features
        
        Simply correct the syntax/parameters of THIS EXACT command to make it work.
        Focus on addressing the specific issue that caused this command to fail.
        
        Your response should be in this format:
        <fixed_command>
        [the corrected command]
        </fixed_command>
        <explanation>
        [brief explanation of what was fixed]
        </explanation>
        """
        
        model_config = {
            "model": getattr(client, 'default_model'),
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": fix_prompt}
            ],
            "max_tokens": 1000,
            "temperature": 0.7
        }
        
        fix_response = BioReasoningHelpers.query_llm(client, fix_prompt, model_config)
        
        # Extract the fixed command from the LLM response
        fixed_match = re.search(r'<fixed_command>(.*?)</fixed_command>', fix_response, re.DOTALL)
        
        if fixed_match:
            fixed_cmd = {
                "action": cmd["action"],
                "action_input": fixed_match.group(1).strip()
            }
        else:
            fixed_cmd = cmd
        
        if doc_filename:
            fix_response = fix_response + f"\n<accessed_documentation>{doc_filename}</accessed_documentation>"
        
        return fixed_cmd, fix_response
    
    @staticmethod
    def get_relevant_api_docs(cmd, error_text, api_docs_dir="api_docs"):
        """Extracts relevant API documentation based on the command and error message."""
        api_context = "RELEVANT API INFORMATION:\nNo specific API documentation could be found for this error."
        doc_filename = None
        
        field_error_match = re.search(r"undefined field (\w+)", error_text)
        endpoint_match = re.search(r"https?://[^/]+/api/(\w+)/", cmd)
        
        if field_error_match and endpoint_match:
            problem_field = field_error_match.group(1)
            endpoint = endpoint_match.group(1)
            
            docs_file = os.path.join(api_docs_dir, f"{endpoint}.txt")
            if os.path.exists(docs_file):
                doc_filename = f"{endpoint}.txt"
                with open(docs_file, 'r') as f:
                    api_docs = f.read()
                
                api_context = f"""
                RELEVANT API DOCUMENTATION:
                
                Error involves undefined field '{problem_field}' in the '{endpoint}' API.
                
                COMPLETE API DOCUMENTATION for {endpoint}:
                {api_docs}
                
                Important: ONLY use fields that are explicitly listed in the documentation above.
                Do not use '{problem_field}' as it is undefined and not available.
                
                If you need to filter by genus/species, you should:
                1. First query the genome API to get genome_ids
                2. Then use those IDs with the genome_feature API
                
                Example pattern:
                ```
                # Get genome_id for Salmonella enterica
                curl -s "https://www.bv-brc.org/api/genome/?eq(genus,Salmonella)&eq(species,enterica)&select(genome_id)&limit(1)"
                
                # Use genome_id to query features
                curl -s "https://www.bv-brc.org/api/genome_feature/?eq(genome_id,GENOME_ID)&select(feature_id,product)&limit(10)"
                ```
                """
        
        return api_context, doc_filename
    
    @staticmethod
    def extract_commands_with_llm(content, client):
        """Use an LLM to extract commands when other parsing methods fail."""
        print("Attempting command extraction using LLM")
        
        prompt = f"""Extract executable commands from the following text. 
Return only the actual commands that should be executed, one per line.
Do not include any explanations or markdown formatting.

Text to extract from:
{content}"""

        model_config = {
            "model": getattr(client, 'default_model'),
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1000,
            "temperature": 0.7
        }
        
        response = BioReasoningHelpers.query_llm(client, prompt, model_config)
        solution = BioReasoningHelpers.remove_think_tags(response)
        
        # Try to parse as JSON and extract commands
        commands = []
        try:
            if solution.strip().startswith('['):
                json_data = json.loads(solution)
                for item in json_data:
                    if isinstance(item, dict) and 'action_input' in item:
                        commands.append({
                            "action": item.get("action", "bash"),
                            "action_input": item["action_input"]
                        })
            elif solution.strip().startswith('{'):
                json_data = json.loads(solution)
                if 'action_input' in json_data:
                    commands.append({
                        "action": json_data.get("action", "bash"),
                        "action_input": json_data["action_input"]
                    })
        except json.JSONDecodeError:
            # If not JSON, treat each line as a command
            lines = [line.strip() for line in solution.split('\n') if line.strip()]
            for line in lines:
                commands.append({
                    "action": "bash",
                    "action_input": line
                })
        
        if commands:
            print(f"LLM extracted {len(commands)} commands")
        else:
            print(f"Errored on Response:\n{response}")
            
        return commands


class ExecutionClassifier:
    """Classifies BV-BRC execution results as success or failure"""
    
    @staticmethod
    def classify_execution(stdout: str, stderr: str) -> Tuple[bool, str]:
        """
        Classify a BV-BRC execution as success or failure.
        Returns (is_success, reason).
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
            
        # Try to parse as JSON to verify valid response
        try:
            json.loads(stdout.strip())
            return True, "Command executed successfully with valid JSON response"
        except json.JSONDecodeError:
            # Non-JSON output might still be valid for some queries
            if len(stdout.strip()) > 0:
                return True, "Command executed successfully with text response"
            return False, "Invalid response format"


class SubsequentStepUpdater:
    """Updates subsequent reasoning steps based on execution results"""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        
    def should_update_subsequent_steps(self, execution_result: Dict, step_index: int, total_steps: int) -> bool:
        """
        Determine if subsequent steps should be updated based on execution result.
        For Phase 1, we use simple heuristics.
        """
        # Only update if execution was successful and we're not at the last step
        if step_index >= total_steps - 1:
            return False
            
        stdout = execution_result.get('stdout', '')
        is_success = execution_result.get('is_success', False)
        
        if not is_success:
            return False
            
        # Update if we got substantial new information
        if len(stdout.strip()) > 50:  # Arbitrary threshold
            return True
            
        return False
        
    def update_subsequent_steps(self, steps: List[ReasoningStep], trigger_index: int, 
                              execution_result: Dict, user_query: str) -> List[ReasoningStep]:
        """
        Update subsequent reasoning steps based on execution results.
        For Phase 1, we implement basic text injection.
        """
        updated_steps = steps.copy()
        
        # Find the next reasoning step after the action
        next_reasoning_index = None
        for i in range(trigger_index + 1, len(steps)):
            if steps[i].step_type == StepType.REASONING:
                next_reasoning_index = i
                break
                
        if next_reasoning_index is not None:
            # Inject execution result into the next reasoning step
            stdout = execution_result.get('stdout', '')
            if stdout.strip():
                # Summarize the output if it's too long
                if len(stdout) > 200:
                    summary = stdout[:200] + "... (truncated)"
                else:
                    summary = stdout
                    
                # Prepend the result to the reasoning step
                original_text = updated_steps[next_reasoning_index].text
                updated_text = f"Based on the query result: {summary}\n\n{original_text}"
                
                updated_steps[next_reasoning_index] = ReasoningStep(
                    step_type=updated_steps[next_reasoning_index].step_type,
                    text=updated_text,
                    start_char=updated_steps[next_reasoning_index].start_char,
                    end_char=updated_steps[next_reasoning_index].end_char,
                    updated=True
                )
                
        return updated_steps


class ExecutionOnlyProcessor:
    """
    Phase 1 implementation of pseudo-chain processing.
    Handles execution-only editing similar to solr_reasoning with retry logic.
    """
    
    def __init__(self, bvbrc_executor: BVBRCShellExecutor, 
                 step_updater: Optional[SubsequentStepUpdater] = None,
                 llm_client=None,
                 max_attempts: int = 3,
                 max_action_retries: int = 3):
        self.bvbrc_executor = bvbrc_executor
        self.step_updater = step_updater or SubsequentStepUpdater(llm_client=llm_client)
        self.classifier = ExecutionClassifier()
        self.llm_client = llm_client
        self.max_attempts = max_attempts
        self.max_action_retries = max_action_retries
        
    def parse_reasoning_steps(self, response_text: str) -> List[ReasoningStep]:
        """
        Parse response text into reasoning steps.
        Identifies reasoning, action, and answer sections.
        """
        steps = []
        
        # Find action blocks
        action_pattern = re.compile(r'<action>\s*(.*?)\s*</action>', re.DOTALL | re.IGNORECASE)
        action_matches = list(action_pattern.finditer(response_text))
        
        current_pos = 0
        
        for action_match in action_matches:
            # Add reasoning text before this action
            if action_match.start() > current_pos:
                reasoning_text = response_text[current_pos:action_match.start()].strip()
                if reasoning_text:
                    steps.append(ReasoningStep(
                        step_type=StepType.REASONING,
                        text=reasoning_text,
                        start_char=current_pos,
                        end_char=action_match.start()
                    ))
            
            # Add the action
            action_content = action_match.group(1).strip()
            steps.append(ReasoningStep(
                step_type=StepType.ACTION,
                text=action_content,
                start_char=action_match.start(),
                end_char=action_match.end()
            ))
            
            current_pos = action_match.end()
        
        # Add any remaining text as reasoning/answer
        if current_pos < len(response_text):
            remaining_text = response_text[current_pos:].strip()
            if remaining_text:
                # Check if this looks like an answer section
                if any(marker in remaining_text.lower() for marker in ['answer:', 'the answer is', 'conclusion:']):
                    step_type = StepType.ANSWER
                else:
                    step_type = StepType.REASONING
                    
                steps.append(ReasoningStep(
                    step_type=step_type,
                    text=remaining_text,
                    start_char=current_pos,
                    end_char=len(response_text)
                ))
        
        return steps
        
    def extract_bvbrc_commands(self, action_text: str) -> List[Dict[str, str]]:
        """
        Extract BV-BRC curl commands from action text using parse_commands logic.
        Handles both JSON format and plain text format.
        """
        commands = []
        
        # Try to parse as JSON first
        try:
            # Handle both single command and array of commands
            if action_text.strip().startswith('['):
                # Array format
                action_data = json.loads(action_text)
                for item in action_data:
                    if isinstance(item, dict) and 'action_input' in item:
                        commands.append({
                            "action": item.get("action", "bash"),
                            "action_input": item['action_input']
                        })
                    elif isinstance(item, str):
                        commands.append({
                            "action": "bash",
                            "action_input": item
                        })
            elif action_text.strip().startswith('{'):
                # Single command format
                action_data = json.loads(action_text)
                if 'action_input' in action_data:
                    commands.append({
                        "action": action_data.get("action", "bash"),
                        "action_input": action_data['action_input']
                    })
            else:
                # Try to extract curl commands directly
                curl_pattern = re.compile(r'curl[^"]*"([^"]*bv-brc\.org[^"]*)"', re.IGNORECASE)
                curl_matches = curl_pattern.findall(action_text)
                commands.extend([{
                    "action": "bash",
                    "action_input": f'curl "{match}"'
                } for match in curl_matches])
                
                # Also try without quotes
                if not commands:
                    curl_pattern2 = re.compile(r'curl\s+([^\s]*bv-brc\.org[^\s]*)', re.IGNORECASE)
                    curl_matches2 = curl_pattern2.findall(action_text)
                    commands.extend([{
                        "action": "bash",
                        "action_input": f'curl {match}'
                    } for match in curl_matches2])
                    
        except json.JSONDecodeError:
            # Fallback to regex extraction
            curl_pattern = re.compile(r'curl[^"]*"([^"]*bv-brc\.org[^"]*)"', re.IGNORECASE)
            curl_matches = curl_pattern.findall(action_text)
            commands.extend([{
                "action": "bash",
                "action_input": f'curl "{match}"'
            } for match in curl_matches])
            
        return commands
        
    def execute_bvbrc_commands_with_retry(self, commands: List[Dict[str, str]], user_query: str = "") -> List[Dict]:
        """Execute BV-BRC commands with retry logic following solr-together.py approach"""
        results = []
        command_index = 0
        command_retry_count = 0  # Track retries for current command
        
        # Execute commands one by one, allowing for updates after each execution
        while command_index < len(commands):
            command_dict = commands[command_index]
            command = command_dict['action_input']
            
            print(f"Executing command {command_index + 1}/{len(commands)}: {command}")
            
            # Execute current command
            stdout, stderr = self.bvbrc_executor.run_query(command)
            is_success, reason = self.classifier.classify_execution(stdout, stderr)
            
            result = {
                'command': command,
                'stdout': stdout,
                'stderr': stderr,
                'is_success': is_success,
                'reason': reason
            }
            results.append(result)
            
            if is_success:
                print(f"✅ Command {command_index + 1} succeeded")
                
                # If command succeeded and there are more commands, update the remaining commands
                if command_index < len(commands) - 1 and self.llm_client:
                    # Classify the output
                    classification = BioReasoningHelpers.classify_last_command_output_with_llm(
                        user_query, command, stdout, self.llm_client
                    )
                    
                    # Update remaining commands based on this result
                    should_update, updated_commands, update_response = BioReasoningHelpers.update_commands_with_llm(
                        user_query, commands, command_index, command, stdout, classification, self.llm_client
                    )
                    
                    if should_update:
                        commands = updated_commands
                        print(f"Commands updated! Next command changed from: {commands[command_index + 1]['action_input']}")
                
                # Move to next command
                command_retry_count = 0  # Reset counter for next command
                command_index += 1
            else:
                print(f"❌ Command {command_index + 1} failed: {reason}")
                
                # Try to fix the failed command using LLM (with retry limit)
                if self.llm_client and command_retry_count < self.max_action_retries:
                    command_retry_count += 1
                    print(f"🔧 Attempting to fix command (retry {command_retry_count}/{self.max_action_retries})")
                    
                    # Classify the failure
                    classification = BioReasoningHelpers.classify_last_command_output_with_llm(
                        user_query, command, stdout if stdout else stderr, self.llm_client
                    )
                    
                    # Fix the failed command
                    fixed_cmd, fix_response = BioReasoningHelpers.fix_failed_command_with_llm(
                        user_query, command_dict, stdout if stdout else stderr, classification, self.llm_client
                    )
                    
                    print(f"🔧 Fixed command: {fixed_cmd['action_input']}")
                    
                    # Update the command in the list and retry it
                    commands[command_index] = fixed_cmd
                    command_dict = fixed_cmd  # Update for next iteration
                    
                    # Don't increment command_index - we'll retry this command
                    continue
                else:
                    if command_retry_count >= self.max_action_retries:
                        print(f"❌ Command {command_index + 1} failed after {self.max_action_retries} retry attempts")
                    # Move to next command if no LLM client or max retries reached
                    command_retry_count = 0  # Reset counter for next command
                    command_index += 1
            
        return results

    def process_response_with_retries(self, response_text: str, user_query: str, 
                                    max_iterations: int = 3) -> str:
        """
        Process a response with retry logic inspired by solr-together.py.
        Implements multi-attempt retries and missing action tag handling.
        """
        previous_attempts = []
        
        for attempt_num in range(self.max_attempts):
            print(f"Processing attempt {attempt_num + 1}/{self.max_attempts}")
            
            current_response = response_text
            
            # Handle missing action tags with retries
            retry_count = 0
            while ("<action>" not in current_response or "</action>" not in current_response) and retry_count < self.max_action_retries:
                print(f"Response missing action tags, attempting LLM extraction (retry {retry_count + 1}/{self.max_action_retries})")
                
                # Try LLM-based command extraction
                if self.llm_client:
                    extracted_commands = BioReasoningHelpers.extract_commands_with_llm(current_response, self.llm_client)
                    if extracted_commands:
                        # Format extracted commands properly
                        formatted_commands = ""
                        for cmd in extracted_commands:
                            formatted_commands += f'{{"action": "{cmd["action"]}", "action_input": "{cmd["action_input"]}"}}\n'
                        
                        # Replace the original response with properly formatted commands
                        current_response = f"<action>\n{formatted_commands}\n</action>"
                        print("Successfully extracted and formatted commands using LLM")
                        break
                
                retry_count += 1
                
            # If we still don't have action tags after retries, record this attempt as failed
            if "<action>" not in current_response or "</action>" not in current_response:
                failure_reason = "No valid action tags found after retries"
                previous_attempts.append(AttemptResult(
                    attempt_number=attempt_num + 1,
                    response_text=current_response,
                    commands=[],
                    execution_results=[],
                    success=False,
                    failure_reason=failure_reason
                ))
                
                print(f"Attempt {attempt_num + 1} failed: {failure_reason}")
                continue
            
            # Parse and execute the response
            steps = self.parse_reasoning_steps(current_response)
            
            # Process each action step
            attempt_success = False
            all_commands = []
            all_execution_results = []
            
            for step_index, step in enumerate(steps):
                if step.step_type != StepType.ACTION:
                    continue
                    
                # Extract BV-BRC commands from this action
                commands = self.extract_bvbrc_commands(step.text)
                if not commands:
                    continue
                    
                all_commands.extend([cmd['action_input'] for cmd in commands])
                
                # Execute commands with retry logic
                execution_results = self.execute_bvbrc_commands_with_retry(commands, user_query)
                all_execution_results.extend(execution_results)
                
                # Check if any command succeeded
                successful_results = [r for r in execution_results if r['is_success']]
                if successful_results:
                    attempt_success = True
                
                # Store execution results in step
                failed_results = [r for r in execution_results if not r['is_success']]
                
                step.execution_result = {
                    'commands': [r['command'] for r in execution_results],
                    'results': execution_results,
                    'success': len(successful_results) > 0
                }
            
            # Record this attempt
            previous_attempts.append(AttemptResult(
                attempt_number=attempt_num + 1,
                response_text=current_response,
                commands=all_commands,
                execution_results=all_execution_results,
                success=attempt_success
            ))
            
            # If attempt was successful, return the processed response
            if attempt_success:
                print(f"Attempt {attempt_num + 1} succeeded!")
                return self.reconstruct_response(steps)
            else:
                print(f"Attempt {attempt_num + 1} failed - no successful command executions")
        
        # If all attempts failed, return the last processed response
        print(f"All {self.max_attempts} attempts failed")
        if previous_attempts:
            last_attempt = previous_attempts[-1]
            steps = self.parse_reasoning_steps(last_attempt.response_text)
            return self.reconstruct_response(steps)
        else:
            return response_text

    def process_response(self, response_text: str, user_query: str, 
                        max_iterations: int = 3) -> str:
        """
        Process a response with pseudo-chain editing following solr_reasoning approach.
        Now includes retry logic for better robustness.
        """
        return self.process_response_with_retries(response_text, user_query, max_iterations)
        
    def reconstruct_response(self, steps: List[ReasoningStep]) -> str:
        """
        Reconstruct the response text from processed steps.
        Includes execution results where available.
        """
        reconstructed_parts = []
        
        for step in steps:
            if step.step_type == StepType.ACTION and step.execution_result:
                # Include the original action
                reconstructed_parts.append(f"<action>\n{step.text}\n</action>")
                
                # Add execution results
                results = step.execution_result['results']
                if results:
                    # Add a summary of execution results
                    reconstructed_parts.append("\n<execution_results>")
                    for result in results:
                        if result['is_success']:
                            # Truncate long outputs
                            stdout = result['stdout']
                            if len(stdout) > 500:
                                stdout = stdout[:500] + "... (truncated)"
                            reconstructed_parts.append(f"Command: {result['command']}")
                            reconstructed_parts.append(f"Result: {stdout}")
                        else:
                            reconstructed_parts.append(f"Command: {result['command']}")
                            reconstructed_parts.append(f"Error: {result['reason']}")
                    reconstructed_parts.append("</execution_results>\n")
            else:
                # Include the step as-is
                reconstructed_parts.append(step.text)
        
        return "\n".join(reconstructed_parts)


# Factory function for easy instantiation
def create_execution_only_processor(bvbrc_timeout: int = 30, llm_client=None,
                                  max_attempts: int = 3, max_action_retries: int = 3) -> ExecutionOnlyProcessor:
    """Create an ExecutionOnlyProcessor with default components and retry logic"""
    bvbrc_executor = BVBRCShellExecutor(timeout=bvbrc_timeout)
    step_updater = SubsequentStepUpdater(llm_client=llm_client)
    
    return ExecutionOnlyProcessor(
        bvbrc_executor=bvbrc_executor,
        step_updater=step_updater,
        llm_client=llm_client,
        max_attempts=max_attempts,
        max_action_retries=max_action_retries
    ) 