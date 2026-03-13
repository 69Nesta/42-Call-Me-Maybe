from .Color import Color
import datetime
from pydantic import BaseModel, Field


class Logger(BaseModel):
    """Lightweight logger for console output.

    Attributes:
        ACTIVE (bool): Whether logging is enabled.
        name (str): Name displayed in log messages.
        color (Color): Color used for the name tag.
    """
    ACTIVE: bool = False
    name: str = Field(..., description='The name of the logger')
    color: Color = Field(..., description='The color of the logger')

    def log(self, message: str, end: str | None = '\n') -> None:
        """Print a debug/info message when logging is enabled.

        Args:
            message (str): Message to print.
            end (str | None): End character appended to the message.
        """
        if (self.ACTIVE):
            print(f'{self._get_format()} {message}', end=end)

    def error(self, message: str, end: str | None = '\n') -> None:
        """Print an error message (always shown).

        Args:
            message (str): Error message to print.
            end (str | None): End character appended to the message.
        """
        print(
            f'{self._get_format()} {Color.RED}[ERROR]{Color.RESET} {message}',
            end=end
        )

    def info(self, message: str, end: str | None = '\n') -> None:
        """Print an informational message (always shown).

        Args:
            message (str): Message to print.
            end (str | None): End character appended to the message.
        """
        print(
            f'{self._get_format()} {Color.RESET} {message}',
            end=end
        )

    def _get_format(self) -> str:
        """Return the formatted prefix used for all log lines.

        Returns:
            str: Formatted prefix including time and colored name tag.
        """
        return f'{Color.GRAY}[{self.get_date_time()}] {self.color}[' +\
               f'{self.name}]{Color.RESET}'

    def get_date_time(self) -> str:
        """Return the current time string used in the log prefix.

        Returns:
            str: Formatted time string (HH:MM:SS).
        """
        now = datetime.datetime.now()
        # return now.strftime("%b %a %d %X")
        return now.strftime("%X")
