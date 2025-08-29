import pandas as pd
import json
import pathlib
from typing import Dict, List, Tuple
import logging
from logging.config import dictConfig
import os
from datetime import datetime

# Configure logging
logging_config = {
    'version': 1,
    'formatters': {
        'default': {
            'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',
            'formatter': 'default'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'data_loader.log',
            'maxBytes': 1000000,
            'backupCount': 3,
            'formatter': 'default'
        }
    },
    'root': {
        'level': 'DEBUG',
        'handlers': ['console', 'file']
    }
}

dictConfig(logging_config)

class DataLoaderException(Exception):
    """Base exception class for data loader"""
    pass

class InvalidDataException(DataLoaderException):
    """Exception for invalid data"""
    pass

class DataLoader:
    """Data loader class for SARA dataset cases and statutes"""
    def __init__(self, data_dir: pathlib.Path, config: Dict):
        """
        Initialize data loader

        Args:
        - data_dir (pathlib.Path): Directory containing SARA dataset
        - config (Dict): Configuration dictionary
        """
        self.data_dir = data_dir
        self.config = config
        self.logger = logging.getLogger(__name__)

    def load_sara_cases(self) -> List[Dict]:
        """
        Load SARA dataset cases

        Returns:
        - List[Dict]: List of case dictionaries
        """
        try:
            cases_file = self.data_dir / 'cases.json'
            with open(cases_file, 'r') as f:
                cases = json.load(f)
            self.logger.info(f'Loaded {len(cases)} cases from {cases_file}')
            return cases
        except FileNotFoundError:
            self.logger.error(f'Cases file not found: {self.data_dir / "cases.json"}')
            raise InvalidDataException('Cases file not found')
        except json.JSONDecodeError as e:
            self.logger.error(f'Failed to parse cases file: {e}')
            raise InvalidDataException('Failed to parse cases file')

    def load_statutes(self) -> List[Dict]:
        """
        Load statutes

        Returns:
        - List[Dict]: List of statute dictionaries
        """
        try:
            statutes_file = self.data_dir / 'statutes.json'
            with open(statutes_file, 'r') as f:
                statutes = json.load(f)
            self.logger.info(f'Loaded {len(statutes)} statutes from {statutes_file}')
            return statutes
        except FileNotFoundError:
            self.logger.error(f'Statutes file not found: {self.data_dir / "statutes.json"}')
            raise InvalidDataException('Statutes file not found')
        except json.JSONDecodeError as e:
            self.logger.error(f'Failed to parse statutes file: {e}')
            raise InvalidDataException('Failed to parse statutes file')

    def validate_data_integrity(self, cases: List[Dict], statutes: List[Dict]) -> bool:
        """
        Validate data integrity

        Args:
        - cases (List[Dict]): List of case dictionaries
        - statutes (List[Dict]): List of statute dictionaries

        Returns:
        - bool: True if data is valid, False otherwise
        """
        try:
            # Check if cases and statutes are lists
            if not isinstance(cases, list) or not isinstance(statutes, list):
                self.logger.error('Cases and statutes must be lists')
                return False

            # Check if cases and statutes are not empty
            if len(cases) == 0 or len(statutes) == 0:
                self.logger.error('Cases and statutes must not be empty')
                return False

            # Check if each case and statute has required fields
            required_fields = ['id', 'text']
            for case in cases:
                if not all(field in case for field in required_fields):
                    self.logger.error(f'Case {case["id"]} is missing required fields')
                    return False
            for statute in statutes:
                if not all(field in statute for field in required_fields):
                    self.logger.error(f'Statute {statute["id"]} is missing required fields')
                    return False

            self.logger.info('Data integrity validated successfully')
            return True
        except Exception as e:
            self.logger.error(f'Failed to validate data integrity: {e}')
            return False

    def create_case_mappings(self, cases: List[Dict], statutes: List[Dict]) -> Dict:
        """
        Create case mappings

        Args:
        - cases (List[Dict]): List of case dictionaries
        - statutes (List[Dict]): List of statute dictionaries

        Returns:
        - Dict: Case mappings dictionary
        """
        try:
            case_mappings = {}
            for case in cases:
                case_id = case['id']
                statute_ids = [statute['id'] for statute in statutes if statute['id'] in case['text']]
                case_mappings[case_id] = statute_ids
            self.logger.info(f'Created case mappings for {len(case_mappings)} cases')
            return case_mappings
        except Exception as e:
            self.logger.error(f'Failed to create case mappings: {e}')
            raise InvalidDataException('Failed to create case mappings')

def main():
    data_dir = pathlib.Path('data')
    config = {
        'data_dir': str(data_dir)
    }
    data_loader = DataLoader(data_dir, config)
    cases = data_loader.load_sara_cases()
    statutes = data_loader.load_statutes()
    if data_loader.validate_data_integrity(cases, statutes):
        case_mappings = data_loader.create_case_mappings(cases, statutes)
        print(case_mappings)

if __name__ == '__main__':
    main()