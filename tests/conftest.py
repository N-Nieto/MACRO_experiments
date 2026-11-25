"""Make `code/lib` importable as `lib` from the tests."""
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
TESTS_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
