import subprocess
import sys
from unittest.mock import Mock

import pytest

from phywae import cli


@pytest.mark.parametrize("command, options, expected", [
    ("train", ["-d", "training.hdf5", "--config", "config.py", "--optimize_data"],
     {"trn_data": "training.hdf5", "config": "config.py", "optimize_data": True}),
    ("encode", ["-m", "model.pt", "-t", "tree data.hdf5"],
     {"model": "model.pt", "tree_data": "tree data.hdf5"}),
    ("decode", ["-m", "model.pt", "-e", "latent.csv"],
     {"model": "model.pt", "encoded_data": "latent.csv"}),
    ("generate", ["-m", "model.pt", "-n", "10", "-o", "samples", "--seed", "42"],
     {"model": "model.pt", "num_samples": 10, "out_prefix": "samples", "seed": 42}),
    ("predict", ["-m", "model.pt", "-d", "trees.hdf5", "-o", "prediction", "-ofmt", "csv"],
     {"model": "model.pt", "data": "trees.hdf5", "out_prefix": "prediction", "out_format": "csv"}),
])
def test_dispatch_passes_parsed_options_to_command(monkeypatch, command, options, expected):
    handler = Mock(return_value=0)
    monkeypatch.setattr(getattr(cli, f"cmd_{command}"), "main", handler)
    original_argv = sys.argv.copy()

    assert cli.main([command, *options]) == 0

    handler.assert_called_once()
    args = handler.call_args.args[0]
    assert args.command == command
    for name, value in expected.items():
        assert getattr(args, name) == value
    assert sys.argv == original_argv


@pytest.mark.parametrize("command", [None, "train", "encode", "decode", "generate", "predict"])
def test_help_exits_without_running_a_workflow(command, capsys):
    with pytest.raises(SystemExit) as error:
        cli.main([command, "--help"] if command else ["--help"])
    assert error.value.code == 0
    output = capsys.readouterr().out
    assert f"usage: phywae{(' ' + command) if command else ''} " in output
    if command is None:
        for name in ("train", "encode", "decode", "generate", "predict"):
            assert name in output


@pytest.mark.parametrize("argv", [
    [], ["unknown"], ["train"], ["encode"], ["decode"], ["generate"], ["predict"],
    ["train", "-d", "training.hdf5", "--unknown-option"],
    ["generate", "-m", "model.pt", "-n", "not-a-number", "-o", "samples"],
])
def test_invalid_invocations_report_usage(argv, capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(argv)
    assert error.value.code == 2
    assert "usage: phywae" in capsys.readouterr().err


def test_python_module_entry_point():
    result = subprocess.run(
        [sys.executable, "-m", "phywae", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "usage: phywae" in result.stdout
    assert "generate" in result.stdout
