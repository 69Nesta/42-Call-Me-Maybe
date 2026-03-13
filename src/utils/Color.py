from enum import Enum


class Color(Enum):
    """Enum mapping symbolic names to ANSI escape codes.

    The enum values are string sequences that can be used to style
    terminal output. Converting the enum to string returns the raw escape
    sequence.
    """
    RESET = "\033[0m"

    BLACK = "\033[30m"
    GRAY = "\033[90m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    ITALIC = "\033[3m"

    def __str__(self) -> str:
        """Return the raw ANSI escape sequence for the enum value.

        Returns:
            str: The escape sequence string for terminal styling.
        """
        return self.value
