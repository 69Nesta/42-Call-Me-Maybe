from .Color import Color
import datetime


class Logger:
    ACTIVE = True

    def __init__(self, name: str, color: Color) -> None:
        self.name: str = name
        self.color: Color = color

    def log(self, message: str) -> None:
        if (self.ACTIVE):
            print(f'{Color.GRAY}[{self.get_date_time()}] {self.color}['
                  f'{self.name}]{Color.RESET} {message}')

    def error(self, message: str) -> None:
        if (self.ACTIVE):
            print(f'{Color.GRAY}[{self.get_date_time()}] {self.color}['
                  f'{self.name}] {Color.RED}[ERROR]{Color.RESET} {message}')

    def get_date_time(self) -> str:
        now = datetime.datetime.now()
        # return now.strftime("%b %a %d %X")
        return now.strftime("%X")
