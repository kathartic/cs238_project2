import sys

from typing import List

def get_filename(args: List[str]) -> str:
    """Returns filename from arguments.

    Args:
      args: List of command-line arguments.
    
    Returns:
      Passed-in filename.
    """
    if len(args) != 2:
        raise Exception("usage: python project2.py <infile>.csv")
    
    in_file_name = args[1]
    filenames = set(["small", "medium", "large"])
    if in_file_name not in filenames:
        raise Exception(f"unsupported filename: {in_file_name}")
    return in_file_name


def main():
    in_file_name = get_filename(sys.argv)


if __name__ == '__main__':
    main()