from pathlib import Path
import sys


# 1. __file__ is the current script. .parents[1] climbs up TWO directories to hit your main project root.
# .resolve() makes sure it uses the absolute, full system path.
ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)

# 2. Check if that root directory is already in Python's search list.
if ROOT_STR not in sys.path:
    # 3. Insert it at position 0 (the absolute highest priority spot).
    sys.path.insert(0, ROOT_STR)