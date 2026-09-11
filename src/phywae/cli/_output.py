def print_output_files(output_files):
    """Print the files produced by a completed CLI command."""
    print("Output files:\n" + "\n".join(str(path) for path in output_files))
