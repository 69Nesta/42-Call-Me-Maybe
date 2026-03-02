# from argparse import ArgumentParser

from .CallMeMaybe import CallMeMaybe
from .utils import Logger, Color


def main() -> None:
    logger = Logger('Main', Color.MAGENTA)
    logger.log('Starting the program...')
    try:
        ai = CallMeMaybe('./data/input/functions_definition.json')

        # ai.prompt('What is the sum of 10 and 10 ?')
    except Exception as e:
        logger.error(str(e))


if __name__ == "__main__":
    main()
