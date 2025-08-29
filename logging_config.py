import logging
import json
import datetime
from typing import Dict, Any
from enum import Enum
from logging.handlers import RotatingFileHandler
from threading import Lock

# Constants
LOG_FILE_NAME = 'neuro_symbolic_tax_reasoning_system.log'
LOG_FILE_SIZE = 10000000  # 10MB
LOG_FILE_BACKUP_COUNT = 5
LOG_LEVEL = logging.DEBUG
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Custom logging levels
class LogLevel(Enum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

# Custom logging exception
class LoggingException(Exception):
    pass

# Custom logging configuration class
class LoggingConfig:
    def __init__(self, log_level: LogLevel = LogLevel.DEBUG, log_file_name: str = LOG_FILE_NAME):
        self.log_level = log_level
        self.log_file_name = log_file_name
        self.logger = None
        self.lock = Lock()

    def setup_logging(self) -> None:
        """
        Set up the logging configuration.

        :return: None
        """
        try:
            self.logger = logging.getLogger('neuro_symbolic_tax_reasoning_system')
            self.logger.setLevel(self.log_level.value)
            handler = RotatingFileHandler(self.log_file_name, maxBytes=LOG_FILE_SIZE, backupCount=LOG_FILE_BACKUP_COUNT)
            formatter = logging.Formatter(LOG_FORMAT)
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        except Exception as e:
            raise LoggingException(f'Failed to set up logging: {str(e)}')

    def log_inference_step(self, step_name: str, step_result: Any) -> None:
        """
        Log an inference step.

        :param step_name: The name of the inference step.
        :param step_result: The result of the inference step.
        :return: None
        """
        try:
            with self.lock:
                self.logger.info(f'Inference step {step_name} completed with result: {json.dumps(step_result)}')
        except Exception as e:
            raise LoggingException(f'Failed to log inference step: {str(e)}')

    def log_error(self, error_message: str, error_details: Dict[str, Any] = None) -> None:
        """
        Log an error.

        :param error_message: The error message.
        :param error_details: Additional error details.
        :return: None
        """
        try:
            with self.lock:
                if error_details:
                    self.logger.error(f'Error: {error_message} - Details: {json.dumps(error_details)}')
                else:
                    self.logger.error(f'Error: {error_message}')
        except Exception as e:
            raise LoggingException(f'Failed to log error: {str(e)}')

    def log_cost_calculation(self, cost: float, calculation_details: Dict[str, Any] = None) -> None:
        """
        Log a cost calculation.

        :param cost: The calculated cost.
        :param calculation_details: Additional calculation details.
        :return: None
        """
        try:
            with self.lock:
                if calculation_details:
                    self.logger.info(f'Cost calculation: {cost} - Details: {json.dumps(calculation_details)}')
                else:
                    self.logger.info(f'Cost calculation: {cost}')
        except Exception as e:
            raise LoggingException(f'Failed to log cost calculation: {str(e)}')

# Helper function to validate log level
def validate_log_level(log_level: str) -> LogLevel:
    try:
        return LogLevel[log_level.upper()]
    except KeyError:
        raise ValueError(f'Invalid log level: {log_level}')

# Helper function to get current timestamp
def get_current_timestamp() -> str:
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# Example usage
if __name__ == '__main__':
    logging_config = LoggingConfig()
    logging_config.setup_logging()
    logging_config.log_inference_step('step1', {'result': 'success'})
    logging_config.log_error('Error message', {'details': 'Error details'})
    logging_config.log_cost_calculation(100.0, {'details': 'Calculation details'})