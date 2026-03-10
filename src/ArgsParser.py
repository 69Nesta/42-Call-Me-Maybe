from argparse import ArgumentParser, Namespace
from pydantic import BaseModel, PrivateAttr
from .utils import Logger, Color
from collections.abc import Sequence
from typing import Any


class ArgsParser(BaseModel):
    _logger: Logger = PrivateAttr()
    _parser: ArgumentParser = PrivateAttr()

    def model_post_init(self, _: Any) -> None:
        self._logger = Logger(name='ArgsParser', color=Color.YELLOW)
        self.register_arguments()

    def register_arguments(self) -> None:
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
            default='./data/input/functions_definition.json'
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
        self._logger.log('Arguments registered successfully.')

    def parse_args(self, args: Sequence[str] | None = None) -> Namespace:
        return self._parser.parse_args(args)
