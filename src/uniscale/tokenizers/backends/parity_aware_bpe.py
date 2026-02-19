"""
Parity-aware BPE backend.

This backend uses the parity-aware-bpe library for training.
Supports: BPE, Super-BPE, PA-BPE, PA-Super-BPE
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

from transformers import PreTrainedTokenizerFast

from uniscale.tokenizers.backends.base import TokenizerBackend


class ParityAwareBPEBackend(TokenizerBackend):
    """
    Backend for parity-aware-bpe library.

    Supports:
    - pa-bpe: Parity-aware BPE (cross-lingual fairness)
    - pa-super-bpe: Parity-aware Super-BPE (two-phase training)

    Note: Both variants require multi-lingual training data for parity computation.
    """

    def get_supported_algorithms(self) -> List[str]:
        return ["pa-bpe", "pa-super-bpe"]

    def _extract_language_data(
        self,
        jsonl_file: str,
        output_path: Path,
        file_prefix: str = "lang"
    ) -> tuple[List[Path], List[str]]:
        """
        Extract language-specific data from JSONL file.

        Args:
            jsonl_file: Path to JSONL file with {"text": ..., "language": ...}
            output_path: Directory to save temporary language files
            file_prefix: Prefix for temporary files (e.g., "train" or "dev")

        Returns:
            Tuple of (list of temp file paths, list of language codes in order)

        Note:
            Assumes data is organized as: [lang1_samples, lang2_samples, ...]
            Each language has N consecutive samples.
        """
        print(f"Extracting language data from {jsonl_file}...")

        # First pass: collect language order and group texts
        lang_texts = {}  # {lang: [texts]}
        lang_order = []  # Preserve language order as they appear

        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Reading {file_prefix} data"):
                data = json.loads(line)
                text = data.get("text", "").strip()
                lang = data.get("language", "misc")

                if lang not in lang_texts:
                    lang_texts[lang] = []
                    lang_order.append(lang)

                if text:
                    lang_texts[lang].append(text)

        print(f"Found {len(lang_order)} languages: {lang_order}")
        for lang in lang_order:
            print(f"  {lang}: {len(lang_texts[lang])} samples")

        # Create temporary files for each language
        temp_files = []
        for lang in lang_order:
            temp_file = output_path / f"{file_prefix}_{lang}.txt"
            with open(temp_file, "w", encoding="utf-8") as f:
                for text in lang_texts[lang]:
                    f.write(text + "\n")
            temp_files.append(temp_file)

        return temp_files, lang_order

    def train(
        self,
        algorithm: str,
        data_file: str,
        vocab_size: int,
        output_dir: str,
        ratio: List[float] = None,
        min_frequency: int = 2,
        window_size: int = 100,
        alpha: int = 2,
        num_workers: int = 1,
        phase1_merges: int = None,
        dev_file: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train a parity-aware tokenizer.

        Args:
            algorithm: "pa-bpe" or "pa-super-bpe"
            data_file: Path to training data (JSONL format)
                Format: {"text": "...", "language": "en"}
                Multi-lingual data organized as: [lang1_samples, lang2_samples, ...]
            vocab_size: Target vocabulary size (number of merge operations)
            output_dir: Directory to save tokenizer
            ratio: Desired compression ratios per language (alternative to dev_file)
            min_frequency: Minimum frequency for merges
            window_size: Window size for moving-window variant (unused for now)
            alpha: Alpha parameter for moving-window variant (unused for now)
            num_workers: Number of worker processes
            phase1_merges: For pa-super-bpe, number of merges in phase1 (default: 60% of vocab_size)
            dev_file: Development data in same JSONL format as data_file
                - Used for parity computation (alternative to ratio)
                - Data organized as: [lang1_samples, lang2_samples, ...]
                - The j-th sample of each language is parallel translation
            **kwargs: Additional parameters

        Returns:
            Dictionary with training metadata

        Usage:
            # PA-BPE with dev file:
            train(algorithm="pa-bpe", data_file="data.jsonl", dev_file="dev.jsonl", vocab_size=32000)

            # PA-BPE with ratio:
            train(algorithm="pa-bpe", data_file="data.jsonl", ratio=[1.5, 1.8, 1.2], vocab_size=32000)

            # PA-Super-BPE:
            train(algorithm="pa-super-bpe", data_file="data.jsonl", dev_file="dev.jsonl", vocab_size=128000)

        Note:
            PA-Super-BPE uses two-step training:
            - Phase 1: Train with whitespace+bytelevel pretokenization (phase1_merges)
            - Phase 2: Load phase1 merges, continue with bytelevel-only to vocab_size

            All data must be in JSONL format with "language" field and have multiple languages.
        """
        if algorithm not in self.get_supported_algorithms():
            raise ValueError(
                f"Algorithm '{algorithm}' not supported by ParityAwareBPEBackend. "
                f"Supported: {self.get_supported_algorithms()}"
            )

        # Validate parity-aware setup
        if dev_file is None and ratio is None:
            raise ValueError(
                f"Parity-aware BPE (algorithm='{algorithm}') requires either:\n"
                f"  --dev_file: Development JSONL with language field\n"
                f"  --ratio: List of desired compression ratios (one per language)\n"
            )

        # Import parity-aware-bpe module
        try:
            from parity_aware_bpe import parity_aware_learn_bpe as learn_bpe_module
        except ImportError:
            raise ImportError(
                "parity-aware-bpe library not found. "
                "Install with: pip install -e tmp/parity-aware-bpe"
            )

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Check if this is a super-bpe variant (requires two-step training)
        is_super_bpe = algorithm in ["pa-super-bpe"]

        if is_super_bpe:
            # Two-step training for SuperBPE
            if phase1_merges is None:
                phase1_merges = int(vocab_size * 0.6)  # Default: 60% in phase1

            print(f"Training SuperBPE tokenizer (two-step process)")
            print(f"  Phase 1: {phase1_merges} merges (whitespace + bytelevel)")
            print(f"  Phase 2: {vocab_size} total merges (bytelevel only)")

            return self._train_super_bpe(
                algorithm=algorithm,
                data_file=data_file,
                vocab_size=vocab_size,
                phase1_merges=phase1_merges,
                output_path=output_path,
                ratio=ratio,
                min_frequency=min_frequency,
                num_workers=num_workers,
                learn_bpe_module=learn_bpe_module,
                dev_file=dev_file,
            )
        else:
            # Single-step training for standard BPE
            print(f"Training {algorithm} tokenizer with vocab_size={vocab_size}")

            # Configure pretokenization
            from tokenizers.pre_tokenizers import Whitespace, ByteLevel
            from tokenizers import pre_tokenizers

            pretokenizer_list = [Whitespace(), ByteLevel(use_regex=False)]
            print("Using pretokenization: whitespace + bytelevel")

            pre_tokenizer = pre_tokenizers.Sequence(pretokenizer_list)
            learn_bpe_module.pre_tokenizer = pre_tokenizer

        # Extract language data from JSONL
        print("Extracting language data from JSONL...")
        train_temp_files, train_langs = self._extract_language_data(
            data_file, output_path, file_prefix="train"
        )

        # Check if multi-lingual (required for parity-aware training)
        num_languages = len(train_langs)
        is_multi_lingual = num_languages > 1

        if not is_multi_lingual:
            raise ValueError(
                f"Parity-aware BPE (algorithm='{algorithm}') requires multi-lingual data, "
                f"but found only 1 language: {train_langs}.\n"
                f"Please provide JSONL with multiple languages."
            )

        print(f"Training mode: {'multi-lingual' if is_multi_lingual else 'single-language'}")
        print(f"Languages: {train_langs}")

        # Prepare input files
        temp_files = train_temp_files
        infiles = [open(f, "r", encoding="utf-8") for f in train_temp_files]
        devfiles = None

        # Load dev files if provided
        if dev_file:
            print(f"Loading dev data from: {dev_file}")
            dev_temp_files, dev_langs = self._extract_language_data(
                dev_file, output_path, file_prefix="dev"
            )

            if dev_langs != train_langs:
                raise ValueError(
                    f"Dev languages {dev_langs} must match training languages {train_langs}"
                )

            temp_files.extend(dev_temp_files)
            devfiles = [open(f, "r", encoding="utf-8") for f in dev_temp_files]

        # Output merges file
        merges_file = output_path / "merges.txt"
        outfile = open(merges_file, "w", encoding="utf-8")

        # Train parity-aware BPE
        try:
            learn_bpe_module.learn_bpe(
                infiles=infiles,
                outfile=outfile,
                devfiles=devfiles,
                num_symbols=vocab_size,
                min_frequency=min_frequency,
                verbose=True,
                is_dict=False,
                total_symbols=False,
                num_global=0,
                ratio=ratio,
                num_workers=num_workers,
                bpe_file=None,
            )
        finally:
            # Close files
            for f in infiles:
                f.close()
            if devfiles:
                for f in devfiles:
                    f.close()
            outfile.close()

        # Clean up temp files
        for temp_file in temp_files:
            if temp_file.exists():
                temp_file.unlink()

        print(f"✓ Saved merges to {merges_file}")

        # Save config
        config = {
            "backend": "parity-aware-bpe",
            "algorithm": algorithm,
            "vocab_size": vocab_size,
            "min_frequency": min_frequency,
        }
        with open(output_path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        return config

    def _train_super_bpe(
        self,
        algorithm: str,
        data_file: str,
        vocab_size: int,
        phase1_merges: int,
        output_path: Path,
        ratio: List[float],
        min_frequency: int,
        num_workers: int,
        learn_bpe_module,
        dev_file: str = None,
    ) -> Dict[str, Any]:
        """
        Train PA-SuperBPE using two-step process.

        Phase 1: Train with whitespace+bytelevel pretokenization
        Phase 2: Load phase1 merges, continue with bytelevel-only
        """
        from tokenizers.pre_tokenizers import Whitespace, ByteLevel
        from tokenizers import pre_tokenizers

        # Extract language data from JSONL
        print("Extracting language data for PA-SuperBPE training...")
        train_temp_files, train_langs = self._extract_language_data(
            data_file, output_path, file_prefix="train_super"
        )

        # Check if multi-lingual (required for PA-SuperBPE)
        num_languages = len(train_langs)
        if num_languages < 2:
            raise ValueError(
                f"PA-SuperBPE requires multi-lingual data, "
                f"but found only 1 language: {train_langs}.\n"
                f"Please provide JSONL with multiple languages."
            )

        print(f"PA-SuperBPE training with {num_languages} languages")
        print(f"Languages: {train_langs}")

        # Track temp files for cleanup
        temp_files = train_temp_files
        training_files = train_temp_files

        # ==================== Phase 1: Standard BPE ====================
        print(f"\n{'='*60}")
        print(f"Phase 1: Training standard BPE ({phase1_merges} merges)")
        print(f"{'='*60}")

        # Set pretokenizer for phase 1 (whitespace + bytelevel)
        pretokenizer_list = [Whitespace(), ByteLevel(use_regex=False)]
        pre_tokenizer = pre_tokenizers.Sequence(pretokenizer_list)
        learn_bpe_module.pre_tokenizer = pre_tokenizer

        # Prepare files for phase 1
        infiles = [open(f, "r", encoding="utf-8") for f in training_files]
        devfiles = None
        if dev_file:
            print(f"Loading dev data from: {dev_file}")
            dev_temp_files, dev_langs = self._extract_language_data(
                dev_file, output_path, file_prefix="dev_super"
            )
            if dev_langs != train_langs:
                raise ValueError(
                    f"Dev languages {dev_langs} must match training languages {train_langs}"
                )
            temp_files.extend(dev_temp_files)
            devfiles = [open(f, "r", encoding="utf-8") for f in dev_temp_files]

        # Output for phase 1
        phase1_merges_file = output_path / "phase1_merges.txt"
        phase1_outfile = open(phase1_merges_file, "w", encoding="utf-8")

        # Train phase 1
        try:
            learn_bpe_module.learn_bpe(
                infiles=infiles,
                outfile=phase1_outfile,
                devfiles=devfiles,
                num_symbols=phase1_merges,
                min_frequency=min_frequency,
                verbose=True,
                is_dict=False,
                total_symbols=False,
                num_global=0,
                ratio=ratio,
                num_workers=num_workers,
                bpe_file=None,
            )
        finally:
            for f in infiles:
                f.close()
            if devfiles:
                for f in devfiles:
                    f.close()
            phase1_outfile.close()

        print(f"✓ Phase 1 completed: {phase1_merges_file}")

        # ==================== Phase 2: Super-BPE ====================
        print(f"\n{'='*60}")
        print(f"Phase 2: Continue with bytelevel-only ({vocab_size} total merges)")
        print(f"{'='*60}")

        # Set pretokenizer for phase 2 (bytelevel only)
        pretokenizer_list = [ByteLevel(use_regex=False)]
        pre_tokenizer = pre_tokenizers.Sequence(pretokenizer_list)
        learn_bpe_module.pre_tokenizer = pre_tokenizer

        # Prepare files for phase 2
        infiles = [open(f, "r", encoding="utf-8") for f in training_files]
        devfiles = None
        if dev_file:
            # Reuse dev files extracted in phase 1
            dev_temp_files, _ = self._extract_language_data(
                dev_file, output_path, file_prefix="dev_super_phase2"
            )
            temp_files.extend(dev_temp_files)
            devfiles = [open(f, "r", encoding="utf-8") for f in dev_temp_files]

        # Load phase1 merges
        bpe_file = open(phase1_merges_file, "r", encoding="utf-8")

        # Output for phase 2 (final merges)
        merges_file = output_path / "merges.txt"
        outfile = open(merges_file, "w", encoding="utf-8")

        # Train phase 2
        try:
            learn_bpe_module.learn_bpe(
                infiles=infiles,
                outfile=outfile,
                devfiles=devfiles,
                num_symbols=vocab_size,
                min_frequency=min_frequency,
                verbose=True,
                is_dict=False,
                total_symbols=False,
                num_global=0,
                ratio=ratio,
                num_workers=num_workers,
                bpe_file=bpe_file,  # Preload phase1 merges
            )
        finally:
            for f in infiles:
                f.close()
            if devfiles:
                for f in devfiles:
                    f.close()
            bpe_file.close()
            outfile.close()

        # Clean up temp files
        for temp_file in temp_files:
            if temp_file.exists():
                temp_file.unlink()

        print(f"✓ Phase 2 completed: {merges_file}")
        print(f"\n✓ SuperBPE training completed!")
        print(f"  - Phase 1 merges: {phase1_merges_file}")
        print(f"  - Final merges: {merges_file}")

        # Save config
        config = {
            "backend": "parity-aware-bpe",
            "algorithm": algorithm,
            "vocab_size": vocab_size,
            "phase1_merges": phase1_merges,
            "min_frequency": min_frequency,
            "two_step": True,
        }
        with open(output_path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        return config

    def export_to_hf(
        self,
        artifacts_dir: str,
        output_dir: str,
        **kwargs
    ) -> PreTrainedTokenizerFast:
        """
        Export parity-aware BPE to HuggingFace format.

        Args:
            artifacts_dir: Directory containing merges.txt
            output_dir: Directory to save HF tokenizer
            **kwargs: Additional parameters

        Returns:
            Loaded HuggingFace tokenizer
        """
        try:
            from parity_aware_bpe.HF_tokenizer import create_huggingface_tokenizer
        except ImportError:
            raise ImportError(
                "parity-aware-bpe library not found. "
                "Install with: pip install -e tmp/parity-aware-bpe"
            )

        artifacts_path = Path(artifacts_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Find merges file
        merges_file = artifacts_path / "merges.txt"
        if not merges_file.exists():
            raise FileNotFoundError(f"Merges file not found: {merges_file}")

        # Convert to HuggingFace tokenizer
        print("Converting parity-aware BPE to HuggingFace format...")
        tokenizer = create_huggingface_tokenizer(
            str(merges_file),
            str(output_path)
        )

        # Save tokenizer
        tokenizer.save_pretrained(str(output_path))

        print(f"✓ Exported HuggingFace tokenizer to {output_path}")

        return tokenizer
