from argparse import ArgumentParser, Namespace
from pydantic import BaseModel, PrivateAttr
from .utils import Logger, Color
from collections.abc import Sequence
from typing import Any


class ArgsParser(BaseModel):
    """Helper that registers and parses CLI arguments.

    Methods:
        register_arguments(): Register supported CLI flags and options.
        parse_args(args): Parse an optional list of arguments and return
            the resulting Namespace.
    """
    _logger: Logger = PrivateAttr()
    _parser: ArgumentParser = PrivateAttr()

    def model_post_init(self, _: Any) -> None:
        self._logger = Logger(name='ArgsParser', color=Color.YELLOW)
        self.register_arguments()

    def register_arguments(self) -> None:
        """Register all supported command-line arguments.

        The arguments include input/output file paths, functions definition
        path, interactive mode toggle, cache directory and model name.
        """
        self._logger.log('Registering command-line arguments...')
        self._parser = ArgumentParser(
            description='Call Me Maybe - A nontrivial modular command-line'
            ' application'
        )

        self._parser.add_argument(
            '--input', '-i',
            help='Function calling tests file path',
            metavar='<input_file>',
            default='./data/input/function_calling_tests.json'
        )
        self._parser.add_argument(
            '--output', '-o',
            help='Output file path for prompts output',
            metavar='<output_file>',
            default='./data/output/prompts_output.json'
        )
        self._parser.add_argument(
            '--functions_definition', '-f',
            help='Functions definition file path',
            metavar='<functions_definition_file>',
            default='./data/input/function_definitions.json'
        )
        self._parser.add_argument(
            '--interactive', '-I',
            help='Run the program in interactive mode',
            action='store_true'
        )
        self._parser.add_argument(
            '--cache_dir', '-c',
            help='Cache directory for the LLM model',
            metavar='<cache_dir>',
            default=None
        )
        self._parser.add_argument(
            '--model_name', '-m',
            help='Name of the LLM model to use',
            metavar='<model_name>',
            default=None
        )
        self._parser.add_argument(
            '--verbose', '-v',
            help='Enable verbose logging',
            action='store_true'
        )
        self._logger.log('Arguments registered successfully.')

    def parse_args(self, args: Sequence[str] | None = None) -> Namespace:
        return self._parser.parse_args(args)
