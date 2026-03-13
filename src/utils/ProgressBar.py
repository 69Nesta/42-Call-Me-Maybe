from pydantic import BaseModel, Field, PrivateAttr
from sys import stdout
from enum import Enum


class StepName(Enum):
    """Names for distinct progress steps shown in the progress bar."""
    NEXT_PROMPT = 'Getting next prompt'
    EXTRACTING_FUNCTION = 'Extracting function'
    EXTRACTING_PARAMETERS = 'Extracting parameters'
    FINISHED = 'Finished'


class ProgressBar(BaseModel):
    """A minimal CLI progress bar used by the test runner.

    Attributes:
        ACTIVE (bool): Whether the progress bar is active.
        total (int): Total number of steps / items.
        current (int): Current progress value.
        length (int): Visual length of the bar in characters.
        current_step_name (StepName): Current descriptive step name.
    """
    ACTIVE: bool = Field(
        default=True,
        description="Whether the progress bar is active"
    )
    total: int = Field(
        0,
        gt=0,
        description="Total number of steps"
    )
    current: int = Field(
        0,
        ge=0,
        description="Current step"
    )
    length: int = Field(
        20,
        gt=0,
        description="Length of the progress bar in characters"
    )
    current_step_name: StepName = Field(
        StepName.EXTRACTING_FUNCTION,
        description="Name of the current step"
    )

    _progress_char: str = PrivateAttr('█')
    _empty_char: str = PrivateAttr('░')
    _max_step_name_length: int = PrivateAttr(
        max(len(step.value) for step in StepName)
    )

    def set_total(self, total: int) -> None:
        """Set the total steps and reset progress to zero.

        Args:
            total (int): Total number of steps (must be > 0).

        Raises:
            ValueError: If total is not greater than zero.
        """
        if total <= 0:
            raise ValueError("Total must be greater than 0")
        self.total = total
        self.current = 0

        self.print()

    def update(self, step: int, step_name: StepName) -> None:
        """Advance the progress bar by a number of steps and set the step name.

        Args:
            step (int): Number of steps to advance (>= 0).
            step_name (StepName): New descriptive step name.

        Raises:
            ValueError: If the step would move beyond total or is negative.
        """
        if step < 0 or self.current + step > self.total:
            raise ValueError(f"Step must be between 0 and {self.total}")
        if step_name not in StepName:
            raise ValueError(f"Invalid step name: {step_name}")
        self.current += step
        self.current_step_name = step_name

        self.print()

    def print(self) -> None:
        """Render the progress bar to stdout if ACTIVE.

        No return value; the progress is written directly to stdout.
        """
        if not self.ACTIVE:
            return
        stdout.write('\r' + str(self))
        stdout.flush()

    def end(self) -> None:
        """Finish the progress bar display (writes a newline).

        No return value; the newline is written directly to stdout.
        """
        if not self.ACTIVE:
            return
        stdout.write('\n')
        stdout.flush()

    def __str__(self) -> str:
        """Return the textual representation of the progress bar.

        Returns:
            str: Visual progress bar with percentage and step name.
        """
        filled_length = int(self.length * self.current // self.total)
        bar = self._progress_char * filled_length + self._empty_char \
            * (self.length - filled_length)
        percent = (self.current / self.total) * 100
        total_str = len(str(self.total // 3))

        progress: str = f'{self.current:{total_str}d}/{self.total}'
        step_name: str = \
            f'{self.current_step_name.value:{self._max_step_name_length}s}'
        return (
            f'|{bar}| {percent:.01f}% - {step_name} - ({progress})'
        )
