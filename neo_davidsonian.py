import re
import typing
import logging
from typing import List, Dict, Tuple
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants and configuration
class Configuration:
    def __init__(self, event_predicate_prefix: str = "event_", argument_separator: str = "_"):
        self.event_predicate_prefix = event_predicate_prefix
        self.argument_separator = argument_separator

class EventStructureError(Exception):
    """Raised when the event structure is invalid."""
    pass

class EventArgumentError(Exception):
    """Raised when an event argument is invalid."""
    pass

class EventPredicate:
    def __init__(self, name: str, arguments: List[str]):
        self.name = name
        self.arguments = arguments

    def __str__(self):
        return f"{self.name}({self.argument_separator.join(self.arguments)})"

class EventStructure:
    def __init__(self, event_predicate: EventPredicate, arguments: Dict[str, str]):
        self.event_predicate = event_predicate
        self.arguments = arguments

    def __str__(self):
        return f"{self.event_predicate} with arguments: {self.arguments}"

def create_event_predicate(name: str, arguments: List[str], config: Configuration) -> EventPredicate:
    """
    Creates an event predicate with the given name and arguments.

    Args:
    name (str): The name of the event predicate.
    arguments (List[str]): The arguments of the event predicate.
    config (Configuration): The configuration object.

    Returns:
    EventPredicate: The created event predicate.

    Raises:
    EventArgumentError: If an argument is invalid.
    """
    if not name:
        raise ValueError("Event predicate name cannot be empty")
    if not arguments:
        raise ValueError("Event predicate arguments cannot be empty")
    for argument in arguments:
        if not argument:
            raise EventArgumentError("Event argument cannot be empty")
    event_predicate_name = f"{config.event_predicate_prefix}{name}"
    return EventPredicate(event_predicate_name, arguments)

def attach_arguments(event_predicate: EventPredicate, arguments: Dict[str, str]) -> EventStructure:
    """
    Attaches the given arguments to the event predicate.

    Args:
    event_predicate (EventPredicate): The event predicate.
    arguments (Dict[str, str]): The arguments to attach.

    Returns:
    EventStructure: The event structure with attached arguments.

    Raises:
    EventStructureError: If the event structure is invalid.
    """
    if not event_predicate:
        raise ValueError("Event predicate cannot be None")
    if not arguments:
        raise ValueError("Arguments cannot be empty")
    for argument_name, argument_value in arguments.items():
        if argument_name not in event_predicate.arguments:
            raise EventStructureError(f"Invalid argument name: {argument_name}")
        if not argument_value:
            raise EventStructureError(f"Argument value cannot be empty: {argument_name}")
    return EventStructure(event_predicate, arguments)

def validate_event_structure(event_structure: EventStructure) -> bool:
    """
    Validates the given event structure.

    Args:
    event_structure (EventStructure): The event structure to validate.

    Returns:
    bool: True if the event structure is valid, False otherwise.
    """
    if not event_structure:
        return False
    if not event_structure.event_predicate:
        return False
    if not event_structure.arguments:
        return False
    for argument_name, argument_value in event_structure.arguments.items():
        if argument_name not in event_structure.event_predicate.arguments:
            return False
        if not argument_value:
            return False
    return True

def convert_to_prolog_fact(event_structure: EventStructure) -> str:
    """
    Converts the given event structure to a Prolog fact.

    Args:
    event_structure (EventStructure): The event structure to convert.

    Returns:
    str: The Prolog fact representation of the event structure.
    """
    if not event_structure:
        raise ValueError("Event structure cannot be None")
    if not validate_event_structure(event_structure):
        raise EventStructureError("Invalid event structure")
    prolog_fact = f"{event_structure.event_predicate.name}({', '.join([f'{argument_name}={argument_value}' for argument_name, argument_value in event_structure.arguments.items()])})"
    return prolog_fact

# Example usage
if __name__ == "__main__":
    config = Configuration()
    event_predicate = create_event_predicate("buy", ["buyer", "seller", "item"], config)
    arguments = {"buyer": "John", "seller": "Alice", "item": "book"}
    event_structure = attach_arguments(event_predicate, arguments)
    logger.info(f"Event structure: {event_structure}")
    if validate_event_structure(event_structure):
        logger.info("Event structure is valid")
    else:
        logger.error("Event structure is invalid")
    prolog_fact = convert_to_prolog_fact(event_structure)
    logger.info(f"Prolog fact: {prolog_fact}")