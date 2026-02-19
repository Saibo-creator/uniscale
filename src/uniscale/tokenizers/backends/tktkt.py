"""
TkTkT backend.

This backend uses the TkTkT (ToKeniser ToolKiT) library for training.
Supports: Various BPE variants, Unigram, and other advanced tokenizers
"""

import json
from pathlib import Path
from typing import List, Dict, Any

from transformers import PreTrainedTokenizerFast

from uniscale.tokenizers.backends.base import TokenizerBackend


class TkTkTBackend(TokenizerBackend):
    """
    Backend for TkTkT (ToKeniser ToolKiT) library.

    Supports:
    - BPE variants: classic-bpe, bpe-dropout, bpe-knockout, picky-bpe, scaffold-bpe, trimmed-bpe
    - Unigram: kudopiece (native implementation)
    - Other: sage, ngram, lzw

    TkTkT provides many more algorithms that can be added as needed.
    """

    def get_supported_algorithms(self) -> List[str]:
        return [
            # BPE variants
            "classic-bpe",
            "bpe-dropout",
            "bpe-knockout",
            "picky-bpe",
            "scaffold-bpe",
            "trimmed-bpe",
            # Unigram
            "kudopiece",
            # Other algorithms
            "sage",
            "ngram",
            "lzw",
        ]

    def train(
        self,
        algorithm: str,
        data_file: str,
        vocab_size: int,
        output_dir: str,
        # BPE-related parameters
        picky_threshold: float = 0.5,
        character_coverage: float = 1.0,
        max_type_length: int = 16,
        # Dropout parameters
        dropout_probability: float = 0.1,
        # KudoPiece parameters
        initial_vocab_size: int = 1_000_000,
        shrinking_factor: float = 0.75,
        num_sub_iterations: int = 2,
        # SaGe parameters
        sage_target_vocab_size: int = None,
        # N-gram parameters
        ngram_n: int = 3,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train a tokenizer using TkTkT library.

        Args:
            algorithm: One of the supported algorithms
            data_file: Path to training data (JSONL)
            vocab_size: Target vocabulary size
            output_dir: Directory to save tokenizer
            picky_threshold: Threshold for PickyBPE (0.0-1.0, default: 0.5)
            character_coverage: Character coverage (default: 1.0)
            max_type_length: Maximum token length (default: 16)
            dropout_probability: Dropout probability for BPE-dropout (default: 0.1)
            initial_vocab_size: Initial vocab size for KudoPiece (default: 1M)
            shrinking_factor: Shrinking factor for KudoPiece (default: 0.75)
            num_sub_iterations: Number of sub-iterations for KudoPiece (default: 2)
            sage_target_vocab_size: Target vocab size for SaGe (if None, uses vocab_size)
            ngram_n: N for N-gram tokenizer (default: 3)
            **kwargs: Algorithm-specific parameters

        Returns:
            Dictionary with training metadata
        """
        if algorithm not in self.get_supported_algorithms():
            raise ValueError(
                f"Algorithm '{algorithm}' not supported by TkTkTBackend. "
                f"Supported: {self.get_supported_algorithms()}"
            )

        # Import TkTkT dependencies
        try:
            from tktkt.preparation.splitters import OnWhitespace
            from tktkt.factories.preprocessors import ModernEnglishPreprocessor, KudoSpaceMarker
        except ImportError as e:
            raise ImportError(
                f"TkTkT library not found or missing dependencies: {e}\n"
                "Install with: pip install -e tmp/TkTkT"
            )

        print(f"Training {algorithm} tokenizer with vocab_size={vocab_size}")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Extract text from JSONL
        texts = []
        print("Loading training data...")
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                text = data.get("text", "").strip()
                if text:
                    texts.append(text)

        print(f"Loaded {len(texts)} training examples")

        # Configure preprocessor (use ModernEnglishPreprocessor for consistency)
        preprocessor = ModernEnglishPreprocessor(marker=KudoSpaceMarker)

        # Train based on algorithm
        if algorithm == "classic-bpe":
            config = self._train_classic_bpe(
                texts, vocab_size, output_path, preprocessor,
                character_coverage, max_type_length
            )

        elif algorithm == "picky-bpe":
            config = self._train_picky_bpe(
                texts, vocab_size, output_path, preprocessor,
                picky_threshold, character_coverage, max_type_length
            )

        elif algorithm == "scaffold-bpe":
            config = self._train_scaffold_bpe(
                texts, vocab_size, output_path, preprocessor,
                character_coverage, max_type_length
            )

        elif algorithm == "trimmed-bpe":
            config = self._train_trimmed_bpe(
                texts, vocab_size, output_path, preprocessor,
                character_coverage, max_type_length
            )

        elif algorithm == "bpe-dropout":
            config = self._train_bpe_dropout(
                texts, vocab_size, output_path, preprocessor,
                dropout_probability, character_coverage, max_type_length
            )

        elif algorithm == "kudopiece":
            config = self._train_kudopiece(
                texts, vocab_size, output_path, preprocessor,
                character_coverage, initial_vocab_size,
                shrinking_factor, num_sub_iterations, max_type_length
            )

        elif algorithm == "sage":
            config = self._train_sage(
                texts, vocab_size, output_path, preprocessor,
                sage_target_vocab_size or vocab_size
            )

        elif algorithm == "ngram":
            config = self._train_ngram(
                texts, vocab_size, output_path, preprocessor, ngram_n
            )

        elif algorithm == "lzw":
            config = self._train_lzw(
                texts, vocab_size, output_path, preprocessor
            )

        else:
            raise NotImplementedError(
                f"Algorithm '{algorithm}' is supported but not yet implemented. "
                "Please implement training logic for this algorithm."
            )

        # Save config
        config.update({
            "backend": "tktkt",
            "algorithm": algorithm,
            "vocab_size": vocab_size,
        })
        with open(output_path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        return config

    def _train_classic_bpe(
        self, texts, vocab_size, output_path, preprocessor,
        character_coverage, max_type_length
    ):
        """Train Classic BPE tokenizer."""
        from tktkt.models.bpe.vocabularisation import BPEVocabulariser, BpeTrainerImplementation

        print("Training Classic BPE...")
        vocabulariser = BPEVocabulariser(
            preprocessor=preprocessor,
            vocab_size=vocab_size,
            implementation=BpeTrainerImplementation.CHIZHOV,
            character_coverage=character_coverage,
            max_token_length=max_type_length,
        )

        # Train
        artifacts = vocabulariser.vocabulariseFromStringIterable(
            texts,
            name_if_not_named=str(output_path / "bpe_cache")
        )

        # Save merges and vocab
        merges = artifacts.getMerges()
        vocab = list(artifacts.getVocabulary())

        # Save merges.txt
        merges_file = output_path / "merges.txt"
        with open(merges_file, "w", encoding="utf-8") as f:
            for merge in merges:
                f.write(" ".join(merge) + "\n")

        # Save vocab.json
        vocab_file = output_path / "vocab.json"
        vocab_dict = {token: idx for idx, token in enumerate(vocab)}
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(vocab_dict, f, indent=2, ensure_ascii=False)

        print(f"✓ Saved merges to {merges_file}")
        print(f"✓ Saved vocab to {vocab_file}")

        return {"type": "bpe"}

    def _train_picky_bpe(
        self, texts, vocab_size, output_path, preprocessor,
        picky_threshold, character_coverage, max_type_length
    ):
        """Train PickyBPE tokenizer."""
        from tktkt.models.bpe.picky import PickyBPEVocabulariser

        print(f"Training PickyBPE (threshold={picky_threshold})...")
        vocabulariser = PickyBPEVocabulariser(
            preprocessor=preprocessor,
            vocab_size=vocab_size,
            picky_threshold=picky_threshold,
            character_coverage=character_coverage,
            max_type_length=max_type_length,
        )

        # Train
        artifacts = vocabulariser.vocabulariseFromStringIterable(
            texts,
            name_if_not_named=str(output_path / "pickybpe_cache")
        )

        # PickyBPE creates events (merges and splits) instead of just merges
        events = artifacts.getEvents()
        vocab = list(artifacts.getVocabulary())

        # Save events.txt (PickyBPE format: + for merge, - for split)
        events_file = output_path / "events.txt"
        with open(events_file, "w", encoding="utf-8") as f:
            for event in events:
                prefix = "+" if event.is_merge else "-"
                f.write(f"{prefix} {event.left} {event.right}\n")

        # For HF export, we need merges.txt with only the merges
        merges_file = output_path / "merges.txt"
        with open(merges_file, "w", encoding="utf-8") as f:
            for event in events:
                if event.is_merge:
                    f.write(f"{event.left} {event.right}\n")

        # Save vocab.json
        vocab_file = output_path / "vocab.json"
        vocab_dict = {token: idx for idx, token in enumerate(vocab)}
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(vocab_dict, f, indent=2, ensure_ascii=False)

        print(f"✓ Saved events to {events_file}")
        print(f"✓ Saved merges to {merges_file}")
        print(f"✓ Saved vocab to {vocab_file}")

        return {"type": "picky-bpe", "picky_threshold": picky_threshold}

    def _train_scaffold_bpe(
        self, texts, vocab_size, output_path, preprocessor,
        character_coverage, max_type_length
    ):
        """Train ScaffoldBPE tokenizer."""
        from tktkt.models.bpe.scaffold import ScaffoldBPEVocabulariser

        print("Training ScaffoldBPE...")
        vocabulariser = ScaffoldBPEVocabulariser(
            preprocessor=preprocessor,
            vocab_size=vocab_size,
            character_coverage=character_coverage,
            max_type_length=max_type_length,
        )

        # Train
        artifacts = vocabulariser.vocabulariseFromStringIterable(
            texts,
            name_if_not_named=str(output_path / "scaffoldbpe_cache")
        )

        # Save merges and vocab
        merges = artifacts.getMerges()
        vocab = list(artifacts.getVocabulary())

        # Save merges.txt
        merges_file = output_path / "merges.txt"
        with open(merges_file, "w", encoding="utf-8") as f:
            for merge in merges:
                f.write(" ".join(merge) + "\n")

        # Save vocab.json
        vocab_file = output_path / "vocab.json"
        vocab_dict = {token: idx for idx, token in enumerate(vocab)}
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(vocab_dict, f, indent=2, ensure_ascii=False)

        print(f"✓ Saved merges to {merges_file}")
        print(f"✓ Saved vocab to {vocab_file}")

        return {"type": "scaffold-bpe"}

    def _train_trimmed_bpe(
        self, texts, vocab_size, output_path, preprocessor,
        character_coverage, max_type_length
    ):
        """Train TrimmedBPE tokenizer."""
        from tktkt.models.bpe.decomposing import TrimmedBPEVocabulariser

        print("Training TrimmedBPE...")
        vocabulariser = TrimmedBPEVocabulariser(
            preprocessor=preprocessor,
            vocab_size=vocab_size,
            character_coverage=character_coverage,
            max_type_length=max_type_length,
        )

        # Train
        artifacts = vocabulariser.vocabulariseFromStringIterable(
            texts,
            name_if_not_named=str(output_path / "trimmedbpe_cache")
        )

        # Save merges and vocab
        merges = artifacts.getMerges()
        vocab = list(artifacts.getVocabulary())

        # Save merges.txt
        merges_file = output_path / "merges.txt"
        with open(merges_file, "w", encoding="utf-8") as f:
            for merge in merges:
                f.write(" ".join(merge) + "\n")

        # Save vocab.json
        vocab_file = output_path / "vocab.json"
        vocab_dict = {token: idx for idx, token in enumerate(vocab)}
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(vocab_dict, f, indent=2, ensure_ascii=False)

        print(f"✓ Saved merges to {merges_file}")
        print(f"✓ Saved vocab to {vocab_file}")

        return {"type": "trimmed-bpe"}

    def _train_bpe_dropout(
        self, texts, vocab_size, output_path, preprocessor,
        dropout_probability, character_coverage, max_type_length
    ):
        """
        Train BPE tokenizer for use with BPE-dropout.

        Note: BPE-dropout is a segmentation algorithm, not a training algorithm.
        We train a standard BPE and use dropout during inference.
        """
        print(f"Training BPE for BPE-dropout (p={dropout_probability})...")
        print("Note: Dropout is applied during inference, not training.")

        # Use standard BPE training
        config = self._train_classic_bpe(
            texts, vocab_size, output_path, preprocessor,
            character_coverage, max_type_length
        )

        config.update({
            "type": "bpe-dropout",
            "dropout_probability": dropout_probability
        })
        return config

    def _train_kudopiece(
        self, texts, vocab_size, output_path, preprocessor,
        character_coverage, initial_vocab_size, shrinking_factor,
        num_sub_iterations, max_type_length
    ):
        """Train KudoPiece (Unigram) tokenizer."""
        from tktkt.models.kudopiece.vocabularisation import (
            KudoPieceVocabulariser,
            KudoPieceArguments
        )

        print("Training KudoPiece (Unigram)...")

        arguments = KudoPieceArguments(
            character_coverage=character_coverage,
            initial_vocab_size=initial_vocab_size,
            maximum_token_length=max_type_length,
            shrinking_factor=shrinking_factor,
            num_sub_iterations=num_sub_iterations,
        )

        vocabulariser = KudoPieceVocabulariser(
            preprocessor=preprocessor,
            final_vocab_size=vocab_size,
            arguments=arguments,
            file_stem=str(output_path / "kudopiece")
        )

        # Train
        artifacts = vocabulariser.vocabulariseFromStringIterable(
            texts,
            name_if_not_named=str(output_path / "kudopiece_cache")
        )

        # KudoPiece stores model file and vocab with log-likelihoods
        model_file = artifacts.getModelFile()
        vocab_with_logp = artifacts.getUnigramLoglikelihoods()

        # Copy model file to output
        import shutil
        target_model = output_path / "tokenizer.model"
        if model_file != target_model:
            shutil.copy(model_file, target_model)

        # Save vocab.json (without log-likelihoods for HF compatibility)
        vocab = list(vocab_with_logp.keys())
        vocab_file = output_path / "vocab.json"
        vocab_dict = {token: idx for idx, token in enumerate(vocab)}
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(vocab_dict, f, indent=2, ensure_ascii=False)

        # Save vocab with log-likelihoods separately
        vocab_logp_file = output_path / "vocab_logp.json"
        with open(vocab_logp_file, "w", encoding="utf-8") as f:
            json.dump(vocab_with_logp, f, indent=2, ensure_ascii=False)

        print(f"✓ Saved model to {target_model}")
        print(f"✓ Saved vocab to {vocab_file}")
        print(f"✓ Saved vocab with log-likelihoods to {vocab_logp_file}")

        return {
            "type": "kudopiece",
            "initial_vocab_size": initial_vocab_size,
            "shrinking_factor": shrinking_factor
        }

    def _train_sage(self, texts, vocab_size, output_path, preprocessor, target_vocab_size):
        """Train SaGe tokenizer."""
        from tktkt.models.sage.vocabularisation import SageVocabulariser

        print("Training SaGe...")
        vocabulariser = SageVocabulariser(
            preprocessor=preprocessor,
            vocab_size=target_vocab_size,
        )

        # Train
        artifacts = vocabulariser.vocabulariseFromStringIterable(
            texts,
            name_if_not_named=str(output_path / "sage_cache")
        )

        # Save vocab
        vocab = list(artifacts.getVocabulary())
        vocab_file = output_path / "vocab.json"
        vocab_dict = {token: idx for idx, token in enumerate(vocab)}
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(vocab_dict, f, indent=2, ensure_ascii=False)

        print(f"✓ Saved vocab to {vocab_file}")
        print("Note: SaGe is a vocabularisation algorithm. Segmentation requires additional setup.")

        return {"type": "sage"}

    def _train_ngram(self, texts, vocab_size, output_path, preprocessor, n):
        """Train N-gram tokenizer."""
        from tktkt.models.ngram.vocabularisation import NgramVocabulariser

        print(f"Training {n}-gram tokenizer...")
        vocabulariser = NgramVocabulariser(
            preprocessor=preprocessor,
            n=n,
        )

        # Train
        artifacts = vocabulariser.vocabulariseFromStringIterable(
            texts,
            name_if_not_named=str(output_path / f"{n}gram_cache")
        )

        # Save vocab
        vocab = list(artifacts.getVocabulary())
        vocab_file = output_path / "vocab.json"
        vocab_dict = {token: idx for idx, token in enumerate(vocab)}
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(vocab_dict, f, indent=2, ensure_ascii=False)

        print(f"✓ Saved vocab to {vocab_file}")

        return {"type": "ngram", "n": n}

    def _train_lzw(self, texts, vocab_size, output_path, preprocessor):
        """Train LZW tokenizer."""
        from tktkt.models.compressive.lzw import LzwVocabulariser

        print("Training LZW tokenizer...")
        vocabulariser = LzwVocabulariser(
            preprocessor=preprocessor,
            max_vocab_size=vocab_size,
        )

        # Train
        artifacts = vocabulariser.vocabulariseFromStringIterable(
            texts,
            name_if_not_named=str(output_path / "lzw_cache")
        )

        # Save vocab
        vocab = list(artifacts.getVocabulary())
        vocab_file = output_path / "vocab.json"
        vocab_dict = {token: idx for idx, token in enumerate(vocab)}
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(vocab_dict, f, indent=2, ensure_ascii=False)

        print(f"✓ Saved vocab to {vocab_file}")
        print("Note: LZW is a compression-based tokenizer with different semantics.")

        return {"type": "lzw"}

    def export_to_hf(
        self,
        artifacts_dir: str,
        output_dir: str,
        **kwargs
    ) -> PreTrainedTokenizerFast:
        """
        Export TkTkT tokenizer to HuggingFace format.

        Args:
            artifacts_dir: Directory containing training artifacts
            output_dir: Directory to save HF tokenizer
            **kwargs: Additional parameters

        Returns:
            Loaded HuggingFace tokenizer
        """
        artifacts_path = Path(artifacts_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load config to determine algorithm
        config_file = artifacts_path / "config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")

        with open(config_file, "r") as f:
            config = json.load(f)

        algorithm = config.get("algorithm")
        tokenizer_type = config.get("type")

        # Handle different tokenizer types
        if tokenizer_type == "kudopiece":
            return self._export_kudopiece_to_hf(artifacts_path, output_path)
        elif tokenizer_type in ["bpe", "picky-bpe", "scaffold-bpe", "trimmed-bpe", "bpe-dropout"]:
            return self._export_bpe_to_hf(artifacts_path, output_path)
        else:
            # For other types, try generic vocab-only export
            return self._export_vocab_only_to_hf(artifacts_path, output_path)

    def _export_bpe_to_hf(self, artifacts_path, output_path):
        """Export BPE-based tokenizer to HuggingFace format."""
        from tokenizers import Tokenizer
        from tokenizers.models import BPE

        merges_file = artifacts_path / "merges.txt"
        vocab_file = artifacts_path / "vocab.json"

        if not merges_file.exists() or not vocab_file.exists():
            raise FileNotFoundError(
                f"Required files not found. Expected:\n"
                f"  - {merges_file}\n"
                f"  - {vocab_file}"
            )

        # Create tokenizer
        tokenizer = Tokenizer(BPE(
            vocab=str(vocab_file),
            merges=str(merges_file)
        ))

        # Wrap in PreTrainedTokenizerFast
        wrapped_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            unk_token="<unk>",
            pad_token="<pad>",
            bos_token="<s>",
            eos_token="</s>",
        )

        # Save
        wrapped_tokenizer.save_pretrained(str(output_path))

        print(f"✓ Exported HuggingFace tokenizer to {output_path}")

        return wrapped_tokenizer

    def _export_kudopiece_to_hf(self, artifacts_path, output_path):
        """Export KudoPiece tokenizer to HuggingFace format."""
        import shutil

        model_file = artifacts_path / "tokenizer.model"
        vocab_file = artifacts_path / "vocab.json"

        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        # Copy model file
        target_model = output_path / "tokenizer.model"
        shutil.copy(model_file, target_model)

        # Create HuggingFace-compatible tokenizer_config.json
        tokenizer_config = {
            "add_bos_token": False,
            "add_eos_token": False,
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "<pad>",
            "unk_token": "<unk>",
            "model_max_length": 2048,
            "padding_side": "right",
            "truncation_side": "right",
            "tokenizer_class": "LlamaTokenizer",  # Use Llama tokenizer class for SPM
        }

        with open(output_path / "tokenizer_config.json", "w") as f:
            json.dump(tokenizer_config, f, indent=2)

        print(f"✓ Exported KudoPiece tokenizer to {output_path}")
        print(f"  Load with: AutoTokenizer.from_pretrained('{output_path}')")

        # Return a dummy tokenizer (actual loading needs LlamaTokenizer)
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(str(output_path))

    def _export_vocab_only_to_hf(self, artifacts_path, output_path):
        """Export vocab-only tokenizer to HuggingFace format."""
        vocab_file = artifacts_path / "vocab.json"

        if not vocab_file.exists():
            raise FileNotFoundError(f"Vocab file not found: {vocab_file}")

        print("Warning: This tokenizer type may not have full HuggingFace support.")
        print("Exporting vocab only.")

        # Copy vocab file
        import shutil
        shutil.copy(vocab_file, output_path / "vocab.json")

        print(f"✓ Exported vocab to {output_path}")

        return None
