from .Color import Color
import datetime
from pydantic import BaseModel, Field


class Logger(BaseModel):
    ACTIVE: bool = False
    name: str = Field(..., description='The name of the logger')
    color: Color = Field(..., description='The color of the logger')

    def log(self, message: str, end: str | None = '\n') -> None:
        if (self.ACTIVE):
            print(f'{self._get_format()} {message}', end=end)

    def error(self, message: str, end: str | None = '\n') -> None:
        print(
            f'{self._get_format()} {Color.RED}[ERROR]{Color.RESET} {message}',
            end=end
        )

    def info(self, message: str, end: str | None = '\n') -> None:
        print(
            f'{self._get_format()} {Color.RESET} {message}',
            end=end
        )

    def _get_format(self) -> str:
        return f'{Color.GRAY}[{self.get_date_time()}] {self.color}[' +\
               f'{self.name}]{Color.RESET}'

    def get_date_time(self) -> str:
        now = datetime.datetime.now()
        # return now.strftime("%b %a %d %X")
        return now.strftime("%X")
