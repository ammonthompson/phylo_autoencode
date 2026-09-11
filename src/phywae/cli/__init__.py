"""Command-line interface for PhyWAE."""

import argparse

from . import cmd_decode, cmd_encode, cmd_generate, cmd_predict, cmd_train


def build_parser():
    parser = argparse.ArgumentParser(
        prog="phywae",
        description="PhyWAE: explore phylogenetic data distributions with autoencoders.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, module, description in (
        ("train", cmd_train, "Train or resume a phylogenetic autoencoder."),
        ("encode", cmd_encode, "Encode trees and auxiliary data into latent coordinates."),
        ("decode", cmd_decode, "Decode saved latent coordinates."),
        ("generate", cmd_generate, "Generate samples with a trained decoder."),
        ("predict", cmd_predict, "Encode and reconstruct trees and auxiliary data."),
    ):
        command = commands.add_parser(name, help=description, description=description)
        module.add_arguments(command)
        command.set_defaults(func=module.main)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)
