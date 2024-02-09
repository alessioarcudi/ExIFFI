import sys
import os 
import logging

def append_dirname(dirname: str, max_levels: int = 10):
    """Append a directory to the system path."""
    # from the current path, go up to max_levels directories to find the directory to append
    path = os.getcwd()
    for _ in range(max_levels):
        path = os.path.dirname(path)
        logging.debug(os.path.basename(path))
        if os.path.basename(path) == dirname and os.path.isdir(path):
            sys.path.append(str(path))
            return
    raise RuntimeError(f"Could not find directory {dirname} in {max_levels} levels")