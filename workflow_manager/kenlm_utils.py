"""
Utilities for generating KenLM language model support files for Pylaia.

Pylaia requires tokens.txt and lexicon.txt files when using a language model.
This module generates these files from the symbols file.
"""

import os
from pathlib import Path
from typing import List, Set


def read_symbols_file(syms_path: str, include_blank: bool = True) -> List[str]:
    """
    Read symbols from Pylaia symbols file.
    
    Format: "symbol index" (one per line)
    Example:
        <ctc> 0
        <space> 1
        A 7
        ...
    
    Args:
        syms_path: Path to symbols file
        include_blank: If True, include <ctc> (blank token). Required for decoder.
    
    Returns:
        List of symbols in order
    """
    symbols = []
    with open(syms_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 1:
                symbol = parts[0]
                # Include <ctc> if requested (needed for decoder)
                if include_blank or symbol != '<ctc>':
                    symbols.append(symbol)
    return symbols


def generate_tokens_file(syms_path: str, tokens_path: str) -> None:
    """
    Generate tokens.txt file from symbols file for Pylaia language model.
    
    The tokens file must include ALL symbols including <ctc> (blank token),
    as the decoder needs to find the blank token in the dictionary.
    
    Args:
        syms_path: Path to Pylaia symbols file
        tokens_path: Path where tokens.txt will be written
    """
    # Include <ctc> as it's required by the decoder
    symbols = read_symbols_file(syms_path, include_blank=True)
    
    # Verify <ctc> is in the list
    if '<ctc>' not in symbols:
        raise ValueError(f"<ctc> token not found in symbols file {syms_path}. First symbol should be <ctc>.")
    
    with open(tokens_path, 'w', encoding='utf-8') as f:
        for symbol in symbols:
            f.write(f"{symbol}\n")
    
    # Verify the file was written correctly
    with open(tokens_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        if first_line != '<ctc>':
            raise ValueError(f"Generated tokens file {tokens_path} does not start with <ctc>. First line: {first_line}")
    
    print(f"Generated tokens file: {tokens_path} ({len(symbols)} tokens, first token: {symbols[0]})")


def generate_lexicon_file(syms_path: str, lexicon_path: str) -> None:
    """
    Generate lexicon.txt file for character-level HTR.
    
    For character-level models, each character maps to itself.
    Format: "word\ttoken1 token2 ..."
    Example:
        A	A
        <space>	<space>
        <ctc>	<ctc>
        hello	h e l l o
    
    Note: <ctc> (blank token) is included as the decoder may need it.
    
    Args:
        syms_path: Path to Pylaia symbols file
        lexicon_path: Path where lexicon.txt will be written
    """
    # Include <ctc> for completeness
    symbols = read_symbols_file(syms_path, include_blank=True)
    
    with open(lexicon_path, 'w', encoding='utf-8') as f:
        # For character-level HTR, each symbol maps to itself
        for symbol in symbols:
            # Write: symbol -> symbol (single token)
            f.write(f"{symbol}\t{symbol}\n")
    
    print(f"Generated lexicon file: {lexicon_path} ({len(symbols)} entries)")


def ensure_kenlm_files(syms_path: str, output_dir: str = None, force_regenerate: bool = False) -> tuple[str, str]:
    """
    Ensure tokens.txt and lexicon.txt exist for KenLM language model.
    
    Creates these files from the symbols file. By default, regenerates them
    to ensure they're up-to-date and include <ctc>.
    
    Args:
        syms_path: Path to Pylaia symbols file
        output_dir: Directory where tokens.txt and lexicon.txt will be created.
                    If None, uses the same directory as syms_path.
        force_regenerate: If True, always regenerate files even if they exist.
                         If False, only regenerate if they don't exist or don't contain <ctc>.
    
    Returns:
        Tuple of (tokens_path, lexicon_path)
    """
    if output_dir is None:
        output_dir = os.path.dirname(syms_path)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tokens_path = str(output_dir / "tokens.txt")
    lexicon_path = str(output_dir / "lexicon.txt")
    
    # Check if tokens file needs regeneration
    needs_regeneration = force_regenerate
    if not needs_regeneration and os.path.exists(tokens_path):
        # Check if tokens file contains <ctc>
        try:
            with open(tokens_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if '<ctc>' not in content:
                    needs_regeneration = True
        except Exception:
            needs_regeneration = True
    
    # Generate tokens file if needed
    if needs_regeneration or not os.path.exists(tokens_path):
        # Delete old file if it exists to ensure clean regeneration
        if os.path.exists(tokens_path):
            os.remove(tokens_path)
        try:
            generate_tokens_file(syms_path, tokens_path)
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to generate tokens file: {e}")
            raise
    
    # Check if lexicon file needs regeneration
    needs_regeneration_lex = force_regenerate
    if not needs_regeneration_lex and os.path.exists(lexicon_path):
        # Check if lexicon file contains <ctc>
        try:
            with open(lexicon_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if '<ctc>' not in content:
                    needs_regeneration_lex = True
        except Exception:
            needs_regeneration_lex = True
    
    # Generate lexicon file if needed
    if needs_regeneration_lex or not os.path.exists(lexicon_path):
        # Delete old file if it exists to ensure clean regeneration
        if os.path.exists(lexicon_path):
            os.remove(lexicon_path)
        generate_lexicon_file(syms_path, lexicon_path)
    
    return tokens_path, lexicon_path

