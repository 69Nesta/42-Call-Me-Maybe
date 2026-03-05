from .Color import Color
import datetime
from pydantic import BaseModel, Field


class Logger(BaseModel):
    ACTIVE: bool = False
    name: str = Field(..., description='The name of the logger')
    color: Color = Field(..., description='The color of the logger')

    def log(self, message: str) -> None:
        if (self.ACTIVE):
            print(f'{Color.GRAY}[{self.get_date_time()}] {self.color}['
                  f'{self.name}]{Color.RESET} {message}')

    def error(self, message: str) -> None:
        print(f'{Color.GRAY}[{self.get_date_time()}] {self.color}['
              f'{self.name}] {Color.RED}[ERROR]{Color.RESET} {message}')

    def get_date_time(self) -> str:
        now = datetime.datetime.now()
        # return now.strftime("%b %a %d %X")
        return now.strftime("%X")
