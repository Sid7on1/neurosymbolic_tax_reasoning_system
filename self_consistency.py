import logging
import numpy as np
from typing import List, Dict
from semantic_parser import SemanticParser
from prolog_executor import PrologExecutor
from data_models import TaxReasoningResult, TaxReasoningInput
from constants import VELOCITY_THRESHOLD, FLOW_THEORY_CONSTANT

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SelfConsistencyChecker:
    """
    Self-consistency checker across multiple reasoning paths.

    This class provides methods to run multiple parsers, check consensus,
    determine confidence, and apply refusal criteria.
    """

    def __init__(self, num_parsers: int, parser_config: Dict):
        """
        Initialize the self-consistency checker.

        Args:
        - num_parsers (int): The number of parsers to run.
        - parser_config (Dict): The configuration for the parsers.
        """
        self.num_parsers = num_parsers
        self.parser_config = parser_config
        self.parsers = [SemanticParser(**parser_config) for _ in range(num_parsers)]
        self.prolog_executor = PrologExecutor()

    def run_multiple_parsers(self, input_data: TaxReasoningInput) -> List[TaxReasoningResult]:
        """
        Run multiple parsers on the input data.

        Args:
        - input_data (TaxReasoningInput): The input data for the parsers.

        Returns:
        - List[TaxReasoningResult]: A list of results from the parsers.
        """
        results = []
        for parser in self.parsers:
            try:
                result = parser.parse(input_data)
                results.append(result)
            except Exception as e:
                logger.error(f"Error running parser: {e}")
                results.append(None)
        return results

    def check_consensus(self, results: List[TaxReasoningResult]) -> bool:
        """
        Check if the results from the parsers are in consensus.

        Args:
        - results (List[TaxReasoningResult]): The results from the parsers.

        Returns:
        - bool: True if the results are in consensus, False otherwise.
        """
        if not results:
            return False
        consensus_result = results[0]
        for result in results[1:]:
            if result is None or result != consensus_result:
                return False
        return True

    def determine_confidence(self, results: List[TaxReasoningResult]) -> float:
        """
        Determine the confidence in the results.

        Args:
        - results (List[TaxReasoningResult]): The results from the parsers.

        Returns:
        - float: The confidence in the results.
        """
        if not results:
            return 0.0
        consensus_result = results[0]
        num_agreeing_results = sum(1 for result in results if result == consensus_result)
        confidence = num_agreeing_results / len(results)
        return confidence

    def apply_refusal_criteria(self, results: List[TaxReasoningResult], confidence: float) -> bool:
        """
        Apply the refusal criteria to the results.

        Args:
        - results (List[TaxReasoningResult]): The results from the parsers.
        - confidence (float): The confidence in the results.

        Returns:
        - bool: True if the results should be refused, False otherwise.
        """
        if confidence < VELOCITY_THRESHOLD:
            return True
        if not self.check_consensus(results):
            return True
        return False

    def execute_prolog(self, input_data: TaxReasoningInput) -> TaxReasoningResult:
        """
        Execute the Prolog program on the input data.

        Args:
        - input_data (TaxReasoningInput): The input data for the Prolog program.

        Returns:
        - TaxReasoningResult: The result from the Prolog program.
        """
        try:
            result = self.prolog_executor.execute(input_data)
            return result
        except Exception as e:
            logger.error(f"Error executing Prolog program: {e}")
            return None

    def calculate_flow_theory_metric(self, results: List[TaxReasoningResult]) -> float:
        """
        Calculate the Flow Theory metric.

        Args:
        - results (List[TaxReasoningResult]): The results from the parsers.

        Returns:
        - float: The Flow Theory metric.
        """
        if not results:
            return 0.0
        consensus_result = results[0]
        num_agreeing_results = sum(1 for result in results if result == consensus_result)
        flow_theory_metric = FLOW_THEORY_CONSTANT * num_agreeing_results / len(results)
        return flow_theory_metric

class SelfConsistencyException(Exception):
    """
    Exception raised when self-consistency checking fails.
    """

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

def main():
    # Create a self-consistency checker
    checker = SelfConsistencyChecker(num_parsers=5, parser_config={"parser_type": "semantic"})

    # Create input data
    input_data = TaxReasoningInput(text="This is a sample input")

    # Run multiple parsers
    results = checker.run_multiple_parsers(input_data)

    # Check consensus
    consensus = checker.check_consensus(results)

    # Determine confidence
    confidence = checker.determine_confidence(results)

    # Apply refusal criteria
    refusal = checker.apply_refusal_criteria(results, confidence)

    # Execute Prolog program
    prolog_result = checker.execute_prolog(input_data)

    # Calculate Flow Theory metric
    flow_theory_metric = checker.calculate_flow_theory_metric(results)

    # Log results
    logger.info(f"Consensus: {consensus}")
    logger.info(f"Confidence: {confidence}")
    logger.info(f"Refusal: {refusal}")
    logger.info(f"Prolog result: {prolog_result}")
    logger.info(f"Flow Theory metric: {flow_theory_metric}")

if __name__ == "__main__":
    main()