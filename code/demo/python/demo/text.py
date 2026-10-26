"""Helpers for text shown to a person."""
MAX_TEXT_LINES = 600  # a circuit can run to thousands of lines, so only the start is shown


def shorten(text: str, limit: int = MAX_TEXT_LINES) -> str:
    """The first `limit` lines of `text`, with a note of how many more there were."""
    lines = text.splitlines()
    if len(lines) <= limit:
        return text
    return "\n".join([*lines[:limit], f"// ... {len(lines) - limit} more lines"])
