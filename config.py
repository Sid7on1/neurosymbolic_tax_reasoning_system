import logging
import os
from typing import Dict, List, Optional
from pydantic import BaseModel, ValidationError, validator
from enum import Enum
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define constants
CONFIG_FILE = 'config.json'
DEFAULT_CONFIG = {
    'models': {
        'prolog': {
            'threshold': 0.5,
            'cost': 0.1
        },
        'llm': {
            'threshold': 0.8,
            'cost': 0.2
        }
    },
    'thresholds': {
        'velocity': 0.3,
        'flow': 0.4
    },
    'cost_parameters': {
        'penalty': 0.1,
        'reward': 0.2
    }
}

class ModelType(str, Enum):
    prolog = 'prolog'
    llm = 'llm'

class ModelSettings(BaseModel):
    threshold: float
    cost: float

    @validator('threshold')
    def validate_threshold(cls, v):
        if v < 0 or v > 1:
            raise ValueError('Threshold must be between 0 and 1')
        return v

    @validator('cost')
    def validate_cost(cls, v):
        if v < 0:
            raise ValueError('Cost must be non-negative')
        return v

class Config(BaseModel):
    models: Dict[ModelType, ModelSettings]
    thresholds: Dict[str, float]
    cost_parameters: Dict[str, float]

    @validator('models')
    def validate_models(cls, v):
        for model, settings in v.items():
            if model not in [ModelType.prolog, ModelType.llm]:
                raise ValueError('Invalid model type')
        return v

    @validator('thresholds')
    def validate_thresholds(cls, v):
        if 'velocity' not in v or 'flow' not in v:
            raise ValueError('Thresholds must include velocity and flow')
        return v

    @validator('cost_parameters')
    def validate_cost_parameters(cls, v):
        if 'penalty' not in v or 'reward' not in v:
            raise ValueError('Cost parameters must include penalty and reward')
        return v

class ConfigManager:
    def __init__(self, config_file: str = CONFIG_FILE):
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self) -> Config:
        try:
            with open(self.config_file, 'r') as f:
                config_data = json.load(f)
            return Config(**config_data)
        except FileNotFoundError:
            logger.warning(f'Config file {self.config_file} not found. Using default config.')
            return Config(**DEFAULT_CONFIG)
        except json.JSONDecodeError:
            logger.error(f'Invalid JSON in config file {self.config_file}. Using default config.')
            return Config(**DEFAULT_CONFIG)
        except ValidationError as e:
            logger.error(f'Invalid config: {e}. Using default config.')
            return Config(**DEFAULT_CONFIG)

    def validate_thresholds(self, thresholds: Dict[str, float]) -> bool:
        if 'velocity' not in thresholds or 'flow' not in thresholds:
            return False
        if thresholds['velocity'] < 0 or thresholds['velocity'] > 1:
            return False
        if thresholds['flow'] < 0 or thresholds['flow'] > 1:
            return False
        return True

    def get_model_settings(self, model_type: ModelType) -> ModelSettings:
        return self.config.models[model_type]

    def get_cost_parameters(self) -> Dict[str, float]:
        return self.config.cost_parameters

def main():
    config_manager = ConfigManager()
    logger.info(config_manager.config)

    # Test validate_thresholds
    thresholds = {'velocity': 0.3, 'flow': 0.4}
    logger.info(f'Valid thresholds: {config_manager.validate_thresholds(thresholds)}')

    # Test get_model_settings
    model_settings = config_manager.get_model_settings(ModelType.prolog)
    logger.info(f'Prolog model settings: {model_settings}')

    # Test get_cost_parameters
    cost_parameters = config_manager.get_cost_parameters()
    logger.info(f'Cost parameters: {cost_parameters}')

if __name__ == '__main__':
    main()