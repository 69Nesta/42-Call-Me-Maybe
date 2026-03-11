from .ArgsParser import ArgsParser
from argparse import Namespace
from .CallMeMaybe import CallMeMaybe
from .CallingTests import CallingTests
from .OutputFile import OutputPrompt
from pydantic import ValidationError
from .utils import Logger, Color, ProgressBar, StepName
import sys


def main() -> None:
    try:
        args_parser: ArgsParser = ArgsParser()
        args: Namespace = args_parser.parse_args(sys.argv[1:])

        logger: Logger = Logger(
            ACTIVE=args.verbose,
            name='Main',
            color=Color.MAGENTA
        )
        logger.log('Starting the program...')

        progress_bar: ProgressBar = ProgressBar(
            total=1,
            current=0,
            length=20,
            current_step_name=StepName.EXTRACTING_FUNCTION,
            ACTIVE=args.verbose
        )

        ai = CallMeMaybe(
            functions_definition_path=str(args.functions_definition),
            output_file_path=str(args.output),
            progress_bar=progress_bar,
            cache_dir=args.cache_dir,
            model_name=args.model_name,
            verbose=args.verbose
        )

        if not args.interactive:
            calling_test = CallingTests(
                file_path=str(args.input),
                prompt_function=ai.prompt,
                progress_bar=progress_bar,
                verbose=args.verbose
            )

            calling_test.run_tests()

            progress_bar.update(0, StepName.FINISHED)
            progress_bar.end()
        else:
            progress_bar.ACTIVE = False
            continue_in_interactive_mode: bool = True
            while continue_in_interactive_mode:
                logger.info('Enter your prompt: ', end='')
                try:
                    output: OutputPrompt = ai.prompt(input(''))
                    logger.info(f'Function used: {output.name}')
                    logger.info('Parameters:')
                    for name, value in output.parameters.items():
                        logger.info(
                            f' - {name}: {value}'
                        )
                except Exception as e:
                    logger.error(f"Error running test: {e}")
                logger.info('---')
                logger.info('Do you want to continue? (y/n): ', end='')
                continue_in_interactive_mode = input('').lower() == 'y'
                if not continue_in_interactive_mode:
                    logger.info('Exiting interactive mode...')
                else:
                    logger.info('---')

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
