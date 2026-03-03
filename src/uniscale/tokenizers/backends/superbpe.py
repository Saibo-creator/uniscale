"""
SuperBPE backend.

Delegates training to the superbpe package via subprocess, using its dedicated
venv (venv/superbpe_venv) which contains a patched HuggingFace tokenizers build.

Two-phase training (mirrors scripts/train_superbpe_tokenizer.sh):
  Phase 1 — train a base BPE tokenizer with a broad pre-tokenizer regex.
  Phase 2 — extend that BPE with SuperBPE using a narrower regex that focuses
             on numbers and punctuation.

Supports algorithm: "superbpe"
"""

import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional

from transformers import PreTrainedTokenizerFast

from uniscale.tokenizers.backends.base import TokenizerBackend


# Paths relative to project root (5 levels up from this file's location)
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
SUPERBPE_PYTHON = _PROJECT_ROOT / "venv" / "superbpe_venv" / "bin" / "python"
SUPERBPE_DIR = _PROJECT_ROOT / "tmp" / "superbpe"

# Named regex presets (same vocabulary as the HuggingFace backend)
REGEX_PRESETS: Dict[str, str] = {
    # Broad patterns — good for Phase 1 (base BPE)
    "apertus": (
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+"
        r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*"
        r"|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    ),
    "gpt4": (
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+"
        r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*"
        r"|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    ),
    "gpt2": (
        r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
    ),
    # Narrow pattern — good for Phase 2 (superbpe extension)
    # Only covers numbers and multi-char punctuation; everything else was already
    # handled by the Phase 1 BPE merges that are inherited.
    "superbpe-phase2": (
        r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]{2,}[\r\n/]*| +(?!\S)"
    ),
    "apertus-phase2": (
        r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+| {2,}(?=\S)|\s+(?!\S)"
    ),
}


def _resolve_regex(pattern: Optional[str]) -> str:
    """Return the raw regex for a named preset or pass through a raw string."""
    if pattern is None:
        return REGEX_PRESETS["gpt4"]
    return REGEX_PRESETS.get(pattern, pattern)


class SuperBPEBackend(TokenizerBackend):
    """
    Backend for the SuperBPE package (tmp/superbpe), run via its own venv.

    Two-phase training:
      Phase 1 — base BPE, saved to <output_dir>/base_bpe/
      Phase 2 — superbpe extension, saved to <output_dir>/

    Parameters for train():
      corpus_dir        directory of .txt files (required; superbpe reads raw text)
      num_bytes         bytes of data to consume (default: 1 GB)
      pre_tokenizer     named preset or raw regex for Phase 1 (default: "gpt4")
      pre_tokenizer_phase2  named preset or raw regex for Phase 2 (default: "superbpe")
      phase1_merges     merges to inherit from Phase 1 into Phase 2
                        (default: 60 % of vocab_size)
    """

    def get_supported_algorithms(self) -> List[str]:
        return ["superbpe"]

    def train(
        self,
        algorithm: str,
        data_file: str,
        vocab_size: int,
        output_dir: str,
        corpus_dir: str = None,
        num_bytes: int = 10 ** 9,
        pre_tokenizer: str = "gpt4",
        pre_tokenizer_phase2: str = "superbpe",
        phase1_merges: int = None,
        superbpe_venv: str = None,
        superbpe_dir: str = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run two-phase SuperBPE training via subprocess.

        Args:
            algorithm:            Must be "superbpe".
            data_file:            Ignored — superbpe reads raw .txt from corpus_dir.
            vocab_size:           Target vocabulary size.
            output_dir:           Root output directory.
            corpus_dir:           Directory of .txt training files (required).
            num_bytes:            Bytes of training data (default: 1 GB).
            pre_tokenizer:        Phase 1 regex — named preset or raw string.
            pre_tokenizer_phase2: Phase 2 regex — named preset or raw string.
            phase1_merges:        Merges inherited from Phase 1 (default: 60 % of vocab_size).
            superbpe_venv:        Path to superbpe venv (overrides module-level default).
            superbpe_dir:         Path to superbpe source dir (overrides module-level default).

        Returns:
            Dict with paths to phase1 and final output directories.
        """
        _tmp_corpus_dir = None  # track temp dir for cleanup
        if corpus_dir is None:
            # If data_file points to a .txt file, wrap it in a temp dir so
            # superbpe's train_tokenizer.py (which requires --corpus_dir) sees
            # exactly that one file.
            if data_file is None:
                raise ValueError(
                    "SuperBPEBackend.train() requires corpus_dir or data_file "
                    "(a path to a .txt file)."
                )
            data_path = Path(data_file).resolve()
            if data_path.suffix != ".txt":
                raise ValueError(
                    f"data_file must be a .txt file when corpus_dir is not set, "
                    f"got: {data_file}"
                )
            if not data_path.exists():
                raise FileNotFoundError(f"data_file not found: {data_path}")
            _tmp_corpus_dir = tempfile.mkdtemp(prefix="superbpe_corpus_")
            (Path(_tmp_corpus_dir) / data_path.name).symlink_to(data_path)
            corpus_dir = _tmp_corpus_dir
            print(f"Using single corpus file: {data_path}")

        # Resolve paths: prefer config-provided values, fall back to module defaults
        python_bin = (
            Path(superbpe_venv) / "bin" / "python" if superbpe_venv
            else SUPERBPE_PYTHON
        )
        train_script = (
            Path(superbpe_dir) / "train_tokenizer.py" if superbpe_dir
            else SUPERBPE_DIR / "train_tokenizer.py"
        )

        if not python_bin.exists():
            raise FileNotFoundError(
                f"SuperBPE venv not found at {python_bin}. "
                "Please set it up with:\n"
                "  python -m venv venv/superbpe_venv\n"
                "  venv/superbpe_venv/bin/pip install -e tmp/superbpe"
            )

        if phase1_merges is None:
            phase1_merges = int(vocab_size * 0.6)

        regex_phase1 = _resolve_regex(pre_tokenizer)
        regex_phase2 = _resolve_regex(pre_tokenizer_phase2)

        output_dir = Path(output_dir)
        phase1_dir = output_dir / "base_bpe"
        phase1_dir.mkdir(parents=True, exist_ok=True)

        # ── Phase 1: base BPE ────────────────────────────────────────────────
        print(f"\n=== SuperBPE Phase 1: base BPE → {phase1_dir} ===")
        subprocess.run(
            [
                str(python_bin), str(train_script),
                "--output_dir", str(phase1_dir),
                "--corpus_dir", str(corpus_dir),
                "--num_bytes", str(num_bytes),
                "--vocab_size", str(vocab_size),
                "--regex_string", regex_phase1,
            ],
            check=True,
        )

        # ── Phase 2: superbpe extension ──────────────────────────────────────
        output_dir.mkdir(parents=True, exist_ok=True)
        # Inherit the first phase1_merges merges from Phase 1
        phase1_merges_src = phase1_dir / "merges.txt"
        phase2_merges_dst = output_dir / "merges.txt"
        with open(phase1_merges_src) as src, open(phase2_merges_dst, "w") as dst:
            for i, line in enumerate(src):
                if i >= phase1_merges:
                    break
                dst.write(line)

        print(f"\n=== SuperBPE Phase 2: extension → {output_dir} ===")
        subprocess.run(
            [
                str(python_bin), str(train_script),
                "--output_dir", str(output_dir),
                "--corpus_dir", str(corpus_dir),
                "--vocab_size", str(vocab_size),
                "--regex_string", regex_phase2,
            ],
            check=True,
        )

        if _tmp_corpus_dir:
            import shutil
            shutil.rmtree(_tmp_corpus_dir, ignore_errors=True)

        return {
            "output_dir": str(output_dir),
            "phase1_dir": str(phase1_dir),
        }

    def export_to_hf(
        self,
        artifacts_dir: str,
        output_dir: str,
        **kwargs,
    ) -> PreTrainedTokenizerFast:
        """
        Export SuperBPE tokenizer to HuggingFace format.

        SuperBPE already produces a standard tokenizer.json, so we just wrap it
        with PreTrainedTokenizerFast and save with a tokenizer_config.json.
        """
        artifacts_path = Path(artifacts_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        tokenizer_json = artifacts_path / "tokenizer.json"
        if not tokenizer_json.exists():
            raise FileNotFoundError(f"tokenizer.json not found in {artifacts_dir}")

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(tokenizer_json),
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
        )
        tokenizer.save_pretrained(str(output_path))

        print(f"✓ Exported HuggingFace tokenizer to {output_path}")
        print(f"  Can now load with: AutoTokenizer.from_pretrained('{output_path}')")

        return tokenizer
