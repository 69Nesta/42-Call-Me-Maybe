from .ArgsParser import ArgsParser, Namespace
from .CallMeMaybe import CallMeMaybe
from .CallingTests import CallingTests
from .OutputFile import OutputPrompt
from pydantic import ValidationError
from .utils import Logger, Color, ProgressBar, StepName
import sys


def main() -> None:
    logger: Logger = Logger(ACTIVE=False, name='Main', color=Color.MAGENTA)
    logger.log('Starting the program...')

    try:
        args_parser: ArgsParser = ArgsParser()
        args: Namespace = args_parser.parse_args(sys.argv[1:])

        progress_bar: ProgressBar = ProgressBar(
            total=1,
            current=0,
            length=20,
            current_step_name=StepName.EXTRACTING_FUNCTION
        )

        ai = CallMeMaybe(
            functions_definition_path=str(args.functions_definition),
            output_file_path=str(args.output),
            progress_bar=progress_bar
        )

        if not args.interactive:
            calling_test = CallingTests(
                file_path=str(args.input),
                prompt_function=ai.prompt,
                progress_bar=progress_bar
            )

            calling_test.run_tests()

            progress_bar.update(0, StepName.FINISHED)
            progress_bar.end()
        else:
            progress_bar.ACTIVE = False
            logger.info('Enter yout prompt: ', end='')
            output: OutputPrompt = ai.prompt(input(''))
            logger.info(f'Function used: {output.name}')
            logger.info('Parameters:')
            for name, value in output.parameters.items():
                logger.info(
                    f' - {name}: {value}'
                )

    except ValidationError as e:
        for error in e.errors():
            if error.get("ctx") and error.get("ctx", {}).get("error"):
                logger.error(f"Error: {error.get('ctx', {}).get('error')}")
            else:
                logger.error(f"Error: {error['msg']}")
    except ValueError as e:
        logger.error(f'Error: {e.__cause__ or e}')


if __name__ == "__main__":
    main()
