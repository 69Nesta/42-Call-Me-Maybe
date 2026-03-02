# from argparse import ArgumentParser

from .CallMeMaybe import CallMeMaybe
from .utils import Logger, Color


def main() -> None:
    logger = Logger('Main', Color.MAGENTA)
    logger.log('Starting the program...')
    try:
        ai = CallMeMaybe('./data/input/functions_definition.json')

        ai.prompt('What is the sum of 2 and 3?')
        # ai.prompt('What is the sum of 10 and 10 ?')
        # ai.prompt('Greet shrek')
        # ai.prompt('Reverse the string \'hello\'')
        ai.prompt(
            'Replace all numbers in \"Hello 34 I\'m '
            '233 years old\" with NUMBERS'
        )
    except Exception as e:
        logger.error(str(e))


if __name__ == "__main__":
    main()
