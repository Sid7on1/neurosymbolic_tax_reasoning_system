# -*- coding: utf-8 -*-

"""
Centralized prompt templates for different parsing strategies.
"""

import logging
import os
from typing import Dict, List, Optional

import jinja2
from pydantic import BaseModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define constants
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), 'templates')
JINJA_ENV = jinja2.Environment(loader=jinja2.FileSystemLoader(TEMPLATE_DIR))

class PromptTemplate(BaseModel):
    """Base class for prompt templates."""
    pass

class ZeroShotPromptTemplate(PromptTemplate):
    """Zero-shot prompt template."""
    template_name: str = 'zero_shot'
    template_path: str = os.path.join(TEMPLATE_DIR, 'zero_shot.jinja2')

    def render(self, input_text: str, output_text: str) -> str:
        """Render the zero-shot prompt template."""
        template = JINJA_ENV.get_template(self.template_path)
        return template.render(input_text=input_text, output_text=output_text)

class FewShotPromptTemplate(PromptTemplate):
    """Few-shot prompt template."""
    template_name: str = 'few_shot'
    template_path: str = os.path.join(TEMPLATE_DIR, 'few_shot.jinja2')

    def render(self, input_text: str, output_text: str, few_shot_examples: List[str]) -> str:
        """Render the few-shot prompt template."""
        template = JINJA_ENV.get_template(self.template_path)
        return template.render(input_text=input_text, output_text=output_text, few_shot_examples=few_shot_examples)

class ParsingInstructionPromptTemplate(PromptTemplate):
    """Parsing instruction prompt template."""
    template_name: str = 'parsing_instruction'
    template_path: str = os.path.join(TEMPLATE_DIR, 'parsing_instruction.jinja2')

    def render(self, input_text: str, output_text: str) -> str:
        """Render the parsing instruction prompt template."""
        template = JINJA_ENV.get_template(self.template_path)
        return template.render(input_text=input_text, output_text=output_text)

class ReasoningPromptTemplate(PromptTemplate):
    """Reasoning prompt template."""
    template_name: str = 'reasoning'
    template_path: str = os.path.join(TEMPLATE_DIR, 'reasoning.jinja2')

    def render(self, input_text: str, output_text: str) -> str:
        """Render the reasoning prompt template."""
        template = JINJA_ENV.get_template(self.template_path)
        return template.render(input_text=input_text, output_text=output_text)

class PromptTemplates:
    """Centralized prompt templates for different parsing strategies."""
    def __init__(self):
        self.templates = {
            'zero_shot': ZeroShotPromptTemplate(),
            'few_shot': FewShotPromptTemplate(),
            'parsing_instruction': ParsingInstructionPromptTemplate(),
            'reasoning': ReasoningPromptTemplate(),
        }

    def get_zero_shot_prompt(self, input_text: str, output_text: str) -> str:
        """Get the zero-shot prompt template."""
        return self.templates['zero_shot'].render(input_text, output_text)

    def get_few_shot_prompt(self, input_text: str, output_text: str, few_shot_examples: List[str]) -> str:
        """Get the few-shot prompt template."""
        return self.templates['few_shot'].render(input_text, output_text, few_shot_examples)

    def get_parsing_instruction_prompt(self, input_text: str, output_text: str) -> str:
        """Get the parsing instruction prompt template."""
        return self.templates['parsing_instruction'].render(input_text, output_text)

    def get_reasoning_prompt(self, input_text: str, output_text: str) -> str:
        """Get the reasoning prompt template."""
        return self.templates['reasoning'].render(input_text, output_text)

# Create an instance of the prompt templates
prompt_templates = PromptTemplates()

# Example usage:
if __name__ == '__main__':
    input_text = 'This is an example input text.'
    output_text = 'This is an example output text.'
    few_shot_examples = ['Example 1', 'Example 2', 'Example 3']

    zero_shot_prompt = prompt_templates.get_zero_shot_prompt(input_text, output_text)
    few_shot_prompt = prompt_templates.get_few_shot_prompt(input_text, output_text, few_shot_examples)
    parsing_instruction_prompt = prompt_templates.get_parsing_instruction_prompt(input_text, output_text)
    reasoning_prompt = prompt_templates.get_reasoning_prompt(input_text, output_text)

    logger.info('Zero-shot prompt: %s', zero_shot_prompt)
    logger.info('Few-shot prompt: %s', few_shot_prompt)
    logger.info('Parsing instruction prompt: %s', parsing_instruction_prompt)
    logger.info('Reasoning prompt: %s', reasoning_prompt)