import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("agents", "environment", "training", "utils", "config"):
    p = os.path.join(_ROOT, sub)
    if p not in sys.path:
        sys.path.insert(0, p)
