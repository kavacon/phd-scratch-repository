"""Syntax errors as a message and a position, for whoever shows them to a person."""
from dataclasses import dataclass
from typing import Optional

from lark.exceptions import UnexpectedCharacters, UnexpectedEOF, UnexpectedToken


@dataclass(frozen=True)
class SourceProblem:
    message: str
    line: Optional[int] = None
    column: Optional[int] = None


def describe_parse_error(error: Exception) -> SourceProblem:
    """A syntax error as a message, with where it happened."""
    if isinstance(error, UnexpectedEOF) or (isinstance(error, UnexpectedToken) and error.token.type == "$END"):
        return SourceProblem("The program ended early. Every program, and every function, ends with a return.")
    if isinstance(error, UnexpectedToken):
        if "INT" in set(error.expected or ()) and "NUMBER" not in set(error.expected or ()):
            message = f"Unexpected '{error.token}': loop bounds must be whole-number literals, not variables or expressions."
        else:
            message = f"Unexpected '{error.token}'."
        return SourceProblem(message, error.line, error.column)
    if isinstance(error, UnexpectedCharacters):
        return SourceProblem(f"Unexpected character '{error.char}'.", error.line, error.column)
    return SourceProblem(str(error))
