"""Move classifier input files.

Usage:
    python -m sc.scripts.move_classifier_inputs
"""

import argparse
import code
import glob
import logging
import os
import typing
from typing import List, Tuple, Set, Dict, Any, Optional, NamedTuple, Iterator

def main() -> None:
    d = 'data/output/classify/input/'
    for p in glob.glob(d + '*.tsv'):
        name = '.'.join(os.path.basename(p).split('.')[:-1])

        new_dir = os.path.join(d, name)
        new_path = os.path.join(new_dir, 'data.tsv')

        os.makedirs(new_dir)
        os.rename(p, new_path)
        print("Moved:")
        print(f" - src: {p}")
        print(f" - dst: {new_path}")
        print()

if __name__ == "__main__":
    main()
