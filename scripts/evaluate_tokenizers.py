"""
Evaluate all trained tokenizers using intrinsic metrics.

This script is a convenience wrapper around the tokenizer evaluation module.
"""

import sys
from pathlib import Path

# Add src to path to enable imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uniscale.evaluation.tokenizer_eval import main

if __name__ == "__main__":
    main()
