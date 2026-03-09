from pydantic import BaseModel, Field, PrivateAttr
from sys import stdout
from enum import Enum


class StepName(Enum):
    NEXT_PROMPT = 'Getting next prompt'
    EXTRACTING_FUNCTION = 'Extracting function'
    EXTRACTING_PARAMETERS = 'Extracting parameters'
    FINISHED = 'Finished'


class ProgressBar(BaseModel):
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
        if total <= 0:
            raise ValueError("Total must be greater than 0")
        self.total = total
        self.current = 0

        self.print()

    def update(self, step: int, step_name: StepName) -> None:
        if step < 0 or self.current + step > self.total:
            raise ValueError(f"Step must be between 0 and {self.total}")
        if step_name not in StepName:
            raise ValueError(f"Invalid step name: {step_name}")
        self.current += step
        self.current_step_name = step_name

        self.print()

    def print(self) -> None:
        if not self.ACTIVE:
            return
        stdout.write('\r' + str(self))
        stdout.flush()

    def end(self) -> None:
        if not self.ACTIVE:
            return
        stdout.write('\n')
        stdout.flush()

    def __str__(self) -> str:
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
