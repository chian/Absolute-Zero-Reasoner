"""
Superprompt processor for bio reasoning tasks.
Uses o3 to enhance prompts based on superprompt templates.
"""

import os
import json
import textwrap
from typing import Optional
from langchain_core.prompts import PromptTemplate
import openai


class SuperpromptProcessor:
    """
    Processes superprompts using o3 to generate enhanced prompts for bio reasoning.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "o1-preview"):
        """
        Initialize the superprompt processor.
        
        Args:
            api_key: OpenAI API key. If None, will try to get from environment.
            model: OpenAI model to use for prompt enhancement.
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.model = model
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # Load superprompt template
        self.superprompt_template = self._load_superprompt_template()
    
    def _load_superprompt_template(self) -> PromptTemplate:
        """Load the bio BVBRC superprompt template."""
        # Try to find the superprompt file
        possible_paths = [
            "bio_bvbrc_superprompt.txt",
            "absolute_zero_reasoner/bio_bvbrc_superprompt.txt",
            os.path.join(os.path.dirname(__file__), "..", "..", "bio_bvbrc_superprompt.txt")
        ]
        
        superprompt_content = None
        for path in possible_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    superprompt_content = f.read()
                break
        
        if superprompt_content is None:
            raise FileNotFoundError("Could not find bio_bvbrc_superprompt.txt file")
        
        return PromptTemplate.from_template(textwrap.dedent(superprompt_content))
    
    def generate_enhanced_prompt(self, user_query: str) -> str:
        """
        Generate an enhanced prompt using o3 based on the superprompt template.
        
        Args:
            user_query: The user's biological query
            
        Returns:
            Enhanced prompt ready for bio reasoning training
        """
        # Format the superprompt with the user query
        formatted_superprompt = self.superprompt_template.format(user_query=user_query)
        
        # Print the input prompt for verification
        print("=" * 80)
        print("INPUT PROMPT TO O3:")
        print("=" * 80)
        print(formatted_superprompt)
        print("=" * 80)
        print()
        
        # Create API call configuration
        messages = [
            {"role": "user", "content": formatted_superprompt}
        ]
        
        # Call o3 to enhance the prompt
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_completion_tokens=32000
        )
        
        enhanced_prompt = response.choices[0].message.content
        
        return enhanced_prompt
    



# Global instance for easy access
_superprompt_processor = None

def get_superprompt_processor() -> SuperpromptProcessor:
    """Get or create the global superprompt processor instance."""
    global _superprompt_processor
    if _superprompt_processor is None:
        _superprompt_processor = SuperpromptProcessor()
    return _superprompt_processor

def generate_enhanced_bio_prompt(user_query: str, use_superprompt: bool = True) -> str:
    """
    Generate an enhanced bio reasoning prompt using o3.
    
    Args:
        user_query: The user's biological query
        use_superprompt: Whether to use o3 superprompt enhancement
        
    Returns:
        Enhanced prompt for bio reasoning
    """
    processor = get_superprompt_processor()
    return processor.generate_enhanced_prompt(user_query) 