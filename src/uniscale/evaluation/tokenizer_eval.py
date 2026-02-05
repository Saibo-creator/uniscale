"""
Evaluate trained tokenizers using tokenizer-intrinsic-evals library.

This script provides comprehensive intrinsic evaluation of tokenizers including:
- Compression rate and fertility metrics
- Information-theoretic metrics (entropy, vocabulary utilization)
- Multilingual fairness (Gini coefficient)
- Morphological alignment (if MorphScore data available)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from tokenizer_analysis import create_analyzer_from_raw_inputs
from tokenizer_analysis.config.language_metadata import LanguageMetadata


def discover_trained_tokenizers(tokenizers_dir: str = "out/tokenizers") -> Dict[str, Dict]:
    """
    Discover all trained tokenizers and create TokEval config.

    Args:
        tokenizers_dir: Directory containing trained tokenizers

    Returns:
        Dictionary in TokEval tokenizer config format
    """
    tokenizers_path = Path(tokenizers_dir)
    if not tokenizers_path.exists():
        raise FileNotFoundError(f"Tokenizers directory not found: {tokenizers_dir}")

    tokenizer_config = {}

    for tok_dir in tokenizers_path.iterdir():
        if not tok_dir.is_dir():
            continue

        tok_name = tok_dir.name

        # All tokenizers are now HF-compatible
        # BPE has tokenizer.json, Unigram has tokenizer.model + tokenizer_config.json
        if (tok_dir / "tokenizer.json").exists() or (tok_dir / "tokenizer_config.json").exists():
            tokenizer_config[tok_name] = {
                "class": "huggingface",
                "path": str(tok_dir.absolute())
            }

    return tokenizer_config


def prepare_language_config(
    eval_data_file: str,
    output_file: str = "tokenizer_eval_lang_config.json",
    max_samples: int = 5000
) -> str:
    """
    Prepare language configuration from evaluation JSONL file.

    Args:
        eval_data_file: Path to JSONL evaluation data
        output_file: Output config file path
        max_samples: Maximum samples to extract per language

    Returns:
        Path to generated config file
    """
    # Load data and group by language
    language_texts = {}

    with open(eval_data_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_samples * 20:  # Rough estimate for early stopping
                break

            data = json.loads(line)
            lang = data.get('language', 'unknown')
            text = data.get('text', '').strip()

            if text:
                if lang not in language_texts:
                    language_texts[lang] = []

                if len(language_texts[lang]) < max_samples:
                    language_texts[lang].append(text)

    # Create temp text files for each language
    eval_data_dir = Path(eval_data_file).parent / "tokenizer_eval_data"
    eval_data_dir.mkdir(exist_ok=True)

    language_config = {"languages": {}}

    for lang, texts in language_texts.items():
        # Create text file
        lang_file = eval_data_dir / f"{lang}.txt"
        with open(lang_file, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text + '\n')

        # Add to config
        language_config["languages"][lang] = str(lang_file.absolute())

    # Save config
    output_path = Path(output_file)
    with open(output_path, 'w') as f:
        json.dump(language_config, f, indent=2)

    print(f"✓ Created language config with {len(language_texts)} languages")
    print(f"✓ Saved to {output_path}")

    return str(output_path.absolute())


def run_tokenizer_evaluation(
    tokenizer_config: Dict[str, Dict],
    language_config_file: str,
    output_dir: str = "out/tokenizer_eval",
    morphscore: bool = False,
    generate_latex: bool = True
):
    """
    Run comprehensive tokenizer evaluation using TokEval.

    Args:
        tokenizer_config: Tokenizer configuration dictionary
        language_config_file: Path to language config file
        output_dir: Output directory for results
        morphscore: Whether to include MorphScore analysis
        generate_latex: Whether to generate LaTeX tables
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save tokenizer config
    tok_config_file = output_path / "tokenizer_config.json"
    with open(tok_config_file, 'w') as f:
        json.dump(tokenizer_config, f, indent=2)

    print(f"\n{'='*60}")
    print("Running Tokenizer Intrinsic Evaluation")
    print(f"{'='*60}")
    print(f"Tokenizers: {len(tokenizer_config)}")
    print(f"Config: {tok_config_file}")
    print(f"Language config: {language_config_file}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    # Load language metadata
    language_metadata = LanguageMetadata(language_config_file)

    # Load and prepare texts
    print("Loading language texts...")
    language_texts = {}
    for lang, lang_path in language_metadata.languages.items():
        with open(lang_path, 'r', encoding='utf-8') as f:
            language_texts[lang] = [line.strip() for line in f if line.strip()]

    # Create analyzer from raw inputs (this handles tokenizer loading automatically)
    print("Creating analyzer...")
    analyzer = create_analyzer_from_raw_inputs(
        tokenizer_configs=tokenizer_config,
        language_texts=language_texts,
        language_metadata=language_metadata,
        plot_save_dir=str(output_path)
    )

    # Run full analysis
    print("\nRunning analysis...")
    results = analyzer.run_analysis()

    # Generate visualizations
    print("\nGenerating visualizations...")
    # Plots are generated as part of run_analysis()

    # Generate LaTeX tables if requested
    if generate_latex:
        print("\nGenerating LaTeX tables...")
        latex_dir = output_path / "latex_tables"
        latex_dir.mkdir(exist_ok=True)
        analyzer.generate_latex_tables(results, output_dir=str(latex_dir))

    # Save results
    results_file = output_path / "analysis_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("Evaluation Complete!")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir}")
    print(f"  - JSON results: {results_file}")
    print(f"  - Plots: {output_path}/*.png")
    if generate_latex:
        print(f"  - LaTeX tables: {latex_dir}/")
    print(f"{'='*60}\n")

    # Print summary
    print("Summary of Key Metrics:")
    print("-" * 60)

    # Fertility and Compression Ratio (have 'global' -> 'mean' structure)
    for metric_name in ['fertility', 'compression_ratio']:
        if metric_name in results and 'per_tokenizer' in results[metric_name]:
            print(f"\n{metric_name.replace('_', ' ').title()}:")
            for tok_name, tok_results in results[metric_name]['per_tokenizer'].items():
                if 'global' in tok_results and 'mean' in tok_results['global']:
                    global_mean = tok_results['global']['mean']
                    print(f"  {tok_name}: {global_mean:.4f}")

    # Vocabulary Utilization (has 'global_utilization' directly)
    if 'vocabulary_utilization' in results and 'per_tokenizer' in results['vocabulary_utilization']:
        print(f"\nVocabulary Utilization:")
        for tok_name, tok_results in results['vocabulary_utilization']['per_tokenizer'].items():
            if 'global_utilization' in tok_results:
                utilization = tok_results['global_utilization']
                used_tokens = tok_results.get('global_used_tokens', 'N/A')
                vocab_size = tok_results.get('global_vocab_size', 'N/A')
                print(f"  {tok_name}: {utilization:.4f} ({used_tokens}/{vocab_size} tokens used)")

    # Gini Coefficient (direct tokenizer mapping)
    if 'tokenizer_fairness_gini' in results:
        print(f"\nMultilingual Fairness (Gini Coefficient):")
        for tok_name, tok_results in results['tokenizer_fairness_gini'].items():
            if isinstance(tok_results, dict) and 'gini_coefficient' in tok_results:
                gini_value = tok_results['gini_coefficient']
                print(f"  {tok_name}: {gini_value:.4f} (lower is more fair)")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained tokenizers with intrinsic metrics"
    )
    parser.add_argument(
        "--tokenizers_dir",
        type=str,
        default="out/tokenizers",
        help="Directory containing trained tokenizers",
    )
    parser.add_argument(
        "--eval_data_file",
        type=str,
        default="data/raw/test_data.jsonl",
        help="Evaluation data file (JSONL with 'text' and 'language' fields)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="out/tokenizer_eval",
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=5000,
        help="Maximum samples per language for evaluation",
    )
    parser.add_argument(
        "--morphscore",
        action="store_true",
        help="Include MorphScore analysis (requires morphscore data)",
    )
    parser.add_argument(
        "--no_latex",
        action="store_true",
        help="Skip LaTeX table generation",
    )

    args = parser.parse_args()

    # Discover tokenizers
    print("Discovering trained tokenizers...")
    tokenizer_config = discover_trained_tokenizers(args.tokenizers_dir)

    if not tokenizer_config:
        print(f"Error: No tokenizers found in {args.tokenizers_dir}")
        return

    print(f"Found {len(tokenizer_config)} tokenizers:")
    for name in tokenizer_config.keys():
        print(f"  - {name}")

    # Prepare language config
    print(f"\nPreparing language configuration from {args.eval_data_file}...")
    language_config_file = prepare_language_config(
        args.eval_data_file,
        max_samples=args.max_samples
    )

    # Run evaluation
    run_tokenizer_evaluation(
        tokenizer_config=tokenizer_config,
        language_config_file=language_config_file,
        output_dir=args.output_dir,
        morphscore=args.morphscore,
        generate_latex=not args.no_latex
    )


if __name__ == "__main__":
    main()
