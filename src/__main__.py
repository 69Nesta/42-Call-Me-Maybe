# from argparse import ArgumentParser
from .CallMeMaybe import CallMeMaybe
from pydantic import ValidationError
from .utils import Logger, Color


def main() -> None:
    logger: Logger = Logger(name='Main', color=Color.MAGENTA)
    logger.log('Starting the program...')
    try:
        ai = CallMeMaybe(
            functions_definition_path='./data/input/functions_definition.json',
            output_file_path='./data/output/prompts_output.json'
        )

        # OutputFile(
        #     file_path='./data/output/prompts_output.json'
        # )

        # ai.prompt(input('Enter your prompt: '))
        # ai.prompt('What is the sum of 3.2 and 1.0?')
        # ai.prompt('What is the sum of 32 and 10?')
        ai.prompt('What is the sum of 444 and 44.4?')
        # ai.prompt('What is the sum of 10 and 10 ?')
        # ai.prompt('Greet shrek')
        # ai.prompt('Greet jhon')

        # ai.prompt('Calculate the square root of 144')
        # ai.prompt('Reverse the string \'hello\'')
        # ai.prompt(
        #     'Replace all numbers in \"Hello 34 I\'m '
        #     '233 years old\" with NUMBERS'
        # )

        # ai.prompt(
        #     'Substitute the word \'cat\' with \'dog\' in \'The cat '
        #     'sat on the mat with another cat\''
        # )

        # print(ai.model.encode('\\"'))
        # print(ai.model.encode("\\'"))
    except ValidationError as e:
        for error in e.errors():
            # location = " -> ".join(str(loc) for loc in error["loc"])
            if error.get("ctx") and error.get("ctx", {}).get("error"):
                logger.error(f"Error: {error.get('ctx', {}).get('error')}")
            else:
                logger.error(f"Error: {error['msg']}")


if __name__ == "__main__":
    main()
