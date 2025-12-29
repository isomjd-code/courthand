#!/usr/bin/env python3
"""
Train a KenLM language model with placeholder-based name injection.

This script:
1. Takes the original corpus
2. Finds surnames, forenames (with Latin cases), and placenames from the database
3. Replaces them with placeholders:
   - PLACEHOLDER_SURNAME
   - PLACEHOLDER_FORENAME_NOMINATIVE, PLACEHOLDER_FORENAME_GENITIVE, etc.
   - PLACEHOLDER_PLACENAME
4. Trains a KenLM model on the placeholder-substituted corpus
5. Parses the .arpa file to find placeholder probabilities
6. For each placeholder:
   - Distributes unigram probability mass among actual names by frequency
   - Identifies top-K most common bigram/trigram contexts containing the placeholder
   - Expands those contexts for ALL actual names with correct class-based probabilities:
     log P("de Smith") = log P("de PLACEHOLDER") + log(freq_Smith / total_freq)
7. Writes the modified .arpa and compiles to binary

This approach is mathematically equivalent to class-based language modeling:
  P(word | context) = P(class | context) × P(word | class)

The top-K context limit keeps model size manageable while ensuring common contexts
have correct probabilities. Rare contexts fall back to unigrams.
"""

import json
import os
import subprocess
import tempfile
import shutil
import re
import math
import sqlite3
from pathlib import Path
import argparse
from typing import Optional, Tuple, List, Dict, Set
from collections import defaultdict

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

# Latin declension cases
LATIN_CASES = ['nominative', 'genitive', 'dative', 'accusative', 'ablative']

# Placeholder tokens
PLACEHOLDER_SURNAME = 'PLACEHOLDER_SURNAME'
PLACEHOLDER_PLACENAME = 'PLACEHOLDER_PLACENAME'
PLACEHOLDER_FORENAME_PREFIX = 'PLACEHOLDER_FORENAME_'

def get_forename_placeholder(case_name: str) -> str:
    """Get placeholder token for a specific forename case."""
    return f"{PLACEHOLDER_FORENAME_PREFIX}{case_name.upper()}"


class NameDatabase:
    """Load and cache name data from the database for placeholder substitution."""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        
        # Lookup sets for detection
        self._surname_set: Set[str] = set()
        self._placename_set: Set[str] = set()
        self._place_abbreviated_set: Set[str] = set()
        self._forename_case_map: Dict[str, str] = {}  # word -> case_name
        
        # Frequency data for probability distribution
        self.surname_frequencies: Dict[str, int] = {}
        self.placename_frequencies: Dict[str, int] = {}
        self.forename_case_frequencies: Dict[str, Dict[str, int]] = {
            case: {} for case in LATIN_CASES
        }
        
        self._load_data()
    
    def _load_data(self):
        """Load all name data from database."""
        print("Loading name data from database...")
        
        self._load_surnames()
        self._load_forenames()
        self._load_placenames()
        
        print(f"  Surnames: {len(self._surname_set)} unique")
        print(f"  Forenames: {len(self._forename_case_map)} unique forms")
        print(f"  Placenames: {len(self._placename_set)} unique")
        for case in LATIN_CASES:
            print(f"    Forename {case}: {len(self.forename_case_frequencies[case])} forms")
    
    def _load_surnames(self):
        """Load surnames with frequencies."""
        try:
            cursor = self.conn.execute("""
                SELECT s.surname, COUNT(DISTINCT ep.entry_id) as frequency
                FROM surnames s
                JOIN person_surnames ps ON s.id = ps.surname_id
                JOIN entry_persons ep ON ps.person_id = ep.person_id
                WHERE s.surname IS NOT NULL AND s.surname != ''
                GROUP BY s.id, s.surname
                HAVING frequency > 0
            """)
            for row in cursor.fetchall():
                surname = row['surname']
                freq = row['frequency']
                self._surname_set.add(surname)
                self.surname_frequencies[surname] = freq
        except sqlite3.OperationalError as e:
            print(f"  Warning: Could not load surnames: {e}")
    
    def _load_forenames(self):
        """Load forename Latin forms with case information and frequencies."""
        try:
            cursor = self.conn.execute("""
                SELECT flf.latin_abbreviated, flf.case_name, COALESCE(f.frequency, 1) as frequency
                FROM forename_latin_forms flf
                JOIN forenames f ON flf.forename_id = f.id
                WHERE flf.latin_abbreviated IS NOT NULL AND flf.latin_abbreviated != ''
                  AND flf.case_name IS NOT NULL
            """)
            for row in cursor.fetchall():
                latin_abbr = row['latin_abbreviated']
                case_name = row['case_name'].lower()
                freq = max(1, row['frequency'])
                
                if case_name not in LATIN_CASES:
                    continue
                
                # Store the mapping from word to case
                # If a word appears in multiple cases, we'll use the first one found
                # (In practice, most forms are unique to their case)
                if latin_abbr not in self._forename_case_map:
                    self._forename_case_map[latin_abbr] = case_name
                
                # Store frequency for this case
                if latin_abbr not in self.forename_case_frequencies[case_name]:
                    self.forename_case_frequencies[case_name][latin_abbr] = freq
                else:
                    # Keep maximum frequency if same form appears multiple times
                    self.forename_case_frequencies[case_name][latin_abbr] = max(
                        self.forename_case_frequencies[case_name][latin_abbr], freq
                    )
        except sqlite3.OperationalError as e:
            print(f"  Warning: Could not load forename_latin_forms: {e}")
    
    def _load_placenames(self):
        """Load placenames with frequencies."""
        # Try loading from places table with frequency
        try:
            cursor = self.conn.execute("""
                SELECT p.name, COALESCE(p.frequency, 1) as frequency
                FROM places p
                WHERE p.name IS NOT NULL AND p.name != ''
            """)
            for row in cursor.fetchall():
                name = row['name']
                freq = max(1, row['frequency'])
                self._placename_set.add(name)
                self.placename_frequencies[name] = freq
        except sqlite3.OperationalError:
            # Fallback: try with entry_places join
            try:
                cursor = self.conn.execute("""
                    SELECT p.name, COUNT(DISTINCT epl.entry_id) as frequency
                    FROM places p
                    LEFT JOIN entry_places epl ON p.id = epl.place_id
                    WHERE p.name IS NOT NULL AND p.name != ''
                    GROUP BY p.id, p.name
                """)
                for row in cursor.fetchall():
                    name = row['name']
                    freq = max(1, row['frequency'])
                    self._placename_set.add(name)
                    self.placename_frequencies[name] = freq
            except sqlite3.OperationalError as e:
                print(f"  Warning: Could not load placenames: {e}")
        
        # Also load place_latin_forms abbreviated forms
        try:
            cursor = self.conn.execute("""
                SELECT plf.latin_abbreviated, p.name, COALESCE(p.frequency, 1) as frequency
                FROM place_latin_forms plf
                JOIN places p ON plf.place_id = p.id
                WHERE plf.latin_abbreviated IS NOT NULL AND plf.latin_abbreviated != ''
            """)
            for row in cursor.fetchall():
                abbr = row['latin_abbreviated']
                freq = max(1, row['frequency'])
                self._place_abbreviated_set.add(abbr)
                # Use the abbreviated form as key
                if abbr not in self.placename_frequencies:
                    self.placename_frequencies[abbr] = freq
        except sqlite3.OperationalError:
            pass
    
    def get_word_type(self, word: str) -> Optional[Tuple[str, Optional[str]]]:
        """
        Determine the type of a word.
        
        Priority order:
        1. Forenames (have case information, most specific)
        2. Placenames (if a word is both a place and surname, use place)
        3. Surnames
        
        Returns:
            Tuple of (type, case) where:
            - type is 'surname', 'forename', 'placename', or None
            - case is the Latin case for forenames, None otherwise
        """
        # Check forenames first (they have case information)
        if word in self._forename_case_map:
            case_name = self._forename_case_map[word]
            return ('forename', case_name)
        
        # Check placenames BEFORE surnames (priority: if both, use placename)
        if word in self._placename_set or word in self._place_abbreviated_set:
            return ('placename', None)
        
        # Check surnames last
        if word in self._surname_set:
            return ('surname', None)
        
        return None
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None


def find_all_step2a_files(base_dir: Path) -> List[Path]:
    """Find all step2a_merged.json files recursively."""
    files = list(base_dir.rglob("step2a_merged.json"))
    print(f"Found {len(files)} step2a_merged.json files")
    return files


def extract_merged_texts(file_paths: List[Path], min_words: int = 2) -> List[str]:
    """Extract merged_text from all JSON files, filtering out texts that are too short."""
    texts = []
    skipped = 0
    filtered = 0
    
    for file_path in tqdm(file_paths, desc="Reading JSON files"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'merged_text' in data and data['merged_text']:
                    text = data['merged_text'].strip()
                    word_count = len(text.split())
                    if word_count >= min_words:
                        texts.append(text)
                    else:
                        filtered += 1
                else:
                    skipped += 1
        except (json.JSONDecodeError, KeyError, IOError) as e:
            print(f"Warning: Failed to read {file_path}: {e}")
            skipped += 1
    
    print(f"Extracted {len(texts)} merged texts ({skipped} files skipped, {filtered} filtered as too short)")
    return texts


def substitute_placeholders(texts: List[str], name_db: NameDatabase) -> Tuple[List[str], Dict[str, int]]:
    """
    Replace names in texts with placeholder tokens.
    
    Returns:
        Tuple of (modified_texts, substitution_counts)
    """
    print("Substituting names with placeholders...")
    
    modified_texts = []
    counts = {
        'surname': 0,
        'placename': 0,
    }
    for case in LATIN_CASES:
        counts[f'forename_{case}'] = 0
    
    for text in tqdm(texts, desc="Processing texts"):
        words = text.split()
        modified_words = []
        
        for word in words:
            # Strip trailing punctuation for lookup, but preserve it
            stripped = word.rstrip(',.;:!?')
            trailing = word[len(stripped):]
            
            word_type = name_db.get_word_type(stripped)
            
            if word_type is None:
                modified_words.append(word)
            elif word_type[0] == 'surname':
                modified_words.append(PLACEHOLDER_SURNAME + trailing)
                counts['surname'] += 1
            elif word_type[0] == 'placename':
                modified_words.append(PLACEHOLDER_PLACENAME + trailing)
                counts['placename'] += 1
            elif word_type[0] == 'forename':
                case_name = word_type[1]
                placeholder = get_forename_placeholder(case_name)
                modified_words.append(placeholder + trailing)
                counts[f'forename_{case_name}'] += 1
            else:
                modified_words.append(word)
        
        modified_texts.append(' '.join(modified_words))
    
    print("\nSubstitution counts:")
    print(f"  Surnames: {counts['surname']:,}")
    print(f"  Placenames: {counts['placename']:,}")
    for case in LATIN_CASES:
        print(f"  Forename {case}: {counts[f'forename_{case}']:,}")
    
    return modified_texts, counts


def prepare_training_text(texts: List[str], output_file: Path) -> None:
    """Prepare training text file for KenLM."""
    print(f"Preparing training text file: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(text + '\n')
    
    line_count = len(texts)
    word_count = sum(len(text.split()) for text in texts)
    print(f"Training text prepared: {line_count:,} lines, {word_count:,} words")


def find_kenlm_binary(binary_name: str) -> Optional[str]:
    """Find KenLM binary in PATH or common locations."""
    try:
        result = subprocess.run(
            ['which', binary_name],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    common_paths = [
        Path.home() / "kenlm" / "build" / "bin" / binary_name,
        Path("/usr/local/bin") / binary_name,
        Path("/usr/bin") / binary_name,
    ]
    
    for path in common_paths:
        if path.exists() and path.is_file():
            return str(path)
    
    return None


def train_kenlm_model(
    training_text_file: Path,
    output_arpa: Path,
    order: int = 3,
    memory: str = "80%",
    lmplz_path: Optional[str] = None
) -> bool:
    """Train a KenLM language model, outputting ARPA format."""
    print(f"\nTraining KenLM model (order={order})...")
    
    if not lmplz_path:
        lmplz_path = find_kenlm_binary('lmplz')
        if not lmplz_path:
            print("Error: lmplz not found. Please install KenLM tools.")
            return False
    
    try:
        cmd = [
            lmplz_path,
            '-o', str(order),
            '-S', memory,
            '--discount_fallback',
            '--text', str(training_text_file),
            '--arpa', str(output_arpa)
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600
        )
        
        if result.returncode != 0:
            print(f"Error building ARPA model:")
            print(result.stderr)
            return False
        
        print(f"✓ ARPA model saved to: {output_arpa}")
        return True
        
    except FileNotFoundError:
        print("Error: lmplz not found.")
        return False
    except subprocess.TimeoutExpired:
        print("Error: Training timed out")
        return False
    except Exception as e:
        print(f"Error during ARPA model building: {e}")
        return False


class ArpaParser:
    """Parse and modify ARPA format language model files."""
    
    def __init__(self, arpa_path: Path):
        self.arpa_path = arpa_path
        self.header_lines: List[str] = []
        self.ngram_counts: Dict[int, int] = {}
        self.unigrams: Dict[str, Tuple[float, float]] = {}  # word -> (log_prob, backoff)
        self.bigrams: List[Tuple[float, str, str, float]] = []  # (log_prob, w1, w2, backoff)
        self.trigrams: List[Tuple[float, str, str, str]] = []  # (log_prob, w1, w2, w3)
        self.higher_ngrams: Dict[int, List[str]] = {}  # order -> raw lines
        
        # Track placeholder contexts for expansion
        # placeholder -> list of (log_prob, context_words, position, backoff)
        # position: 0 = placeholder is first word, 1 = second, 2 = third
        self.placeholder_bigrams: Dict[str, List[Tuple[float, Tuple[str, ...], int, float]]] = defaultdict(list)
        self.placeholder_trigrams: Dict[str, List[Tuple[float, Tuple[str, ...], int]]] = defaultdict(list)
        
    def parse(self):
        """Parse the ARPA file."""
        print(f"Parsing ARPA file: {self.arpa_path}")
        
        with open(self.arpa_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        current_section = None
        current_order = 0
        
        for line in tqdm(lines, desc="Parsing ARPA"):
            line = line.rstrip('\n')
            
            if line.startswith('\\data\\'):
                current_section = 'header'
                self.header_lines.append(line)
                continue
            
            if line.startswith('ngram '):
                # Parse ngram count: "ngram 1=33275"
                match = re.match(r'ngram (\d+)=(\d+)', line)
                if match:
                    order = int(match.group(1))
                    count = int(match.group(2))
                    self.ngram_counts[order] = count
                self.header_lines.append(line)
                continue
            
            if line.startswith('\\') and '-grams:' in line:
                # Parse section header: "\1-grams:"
                match = re.match(r'\\(\d+)-grams:', line)
                if match:
                    current_order = int(match.group(1))
                    current_section = f'{current_order}-grams'
                continue
            
            if line.startswith('\\end\\'):
                current_section = 'end'
                continue
            
            if not line.strip():
                continue
            
            # Parse n-gram entries
            if current_section == '1-grams':
                self._parse_unigram(line)
            elif current_section == '2-grams':
                self._parse_bigram(line)
            elif current_section == '3-grams':
                self._parse_trigram(line)
            elif current_order > 3:
                if current_order not in self.higher_ngrams:
                    self.higher_ngrams[current_order] = []
                self.higher_ngrams[current_order].append(line)
        
        print(f"  Parsed {len(self.unigrams):,} unigrams")
        print(f"  Parsed {len(self.bigrams):,} bigrams")
        print(f"  Parsed {len(self.trigrams):,} trigrams")
    
    def _parse_unigram(self, line: str):
        """Parse a unigram line: log_prob word backoff"""
        parts = line.split('\t')
        if len(parts) >= 2:
            log_prob = float(parts[0])
            word = parts[1]
            backoff = float(parts[2]) if len(parts) > 2 else 0.0
            self.unigrams[word] = (log_prob, backoff)
    
    def _parse_bigram(self, line: str):
        """Parse a bigram line: log_prob word1 word2 backoff"""
        parts = line.split('\t')
        if len(parts) >= 2:
            log_prob = float(parts[0])
            words = parts[1].split()
            if len(words) >= 2:
                backoff = float(parts[2]) if len(parts) > 2 else 0.0
                self.bigrams.append((log_prob, words[0], words[1], backoff))
                
                # Track placeholder contexts
                w1, w2 = words[0], words[1]
                if self._is_placeholder(w1):
                    # Placeholder is first word: "PLACEHOLDER w2"
                    self.placeholder_bigrams[w1].append((log_prob, (w2,), 0, backoff))
                if self._is_placeholder(w2):
                    # Placeholder is second word: "w1 PLACEHOLDER"
                    self.placeholder_bigrams[w2].append((log_prob, (w1,), 1, backoff))
    
    def _parse_trigram(self, line: str):
        """Parse a trigram line: log_prob word1 word2 word3"""
        parts = line.split('\t')
        if len(parts) >= 2:
            log_prob = float(parts[0])
            words = parts[1].split()
            if len(words) >= 3:
                self.trigrams.append((log_prob, words[0], words[1], words[2]))
                
                # Track placeholder contexts
                w1, w2, w3 = words[0], words[1], words[2]
                if self._is_placeholder(w1):
                    # "PLACEHOLDER w2 w3"
                    self.placeholder_trigrams[w1].append((log_prob, (w2, w3), 0))
                if self._is_placeholder(w2):
                    # "w1 PLACEHOLDER w3"
                    self.placeholder_trigrams[w2].append((log_prob, (w1, w3), 1))
                if self._is_placeholder(w3):
                    # "w1 w2 PLACEHOLDER"
                    self.placeholder_trigrams[w3].append((log_prob, (w1, w2), 2))
    
    def _is_placeholder(self, word: str) -> bool:
        """Check if a word is a placeholder token."""
        return word.startswith('PLACEHOLDER_')
    
    def get_placeholder_probabilities(self) -> Dict[str, float]:
        """Extract log probabilities for all placeholder tokens."""
        placeholders = {}
        
        # Surname placeholder
        if PLACEHOLDER_SURNAME in self.unigrams:
            placeholders[PLACEHOLDER_SURNAME] = self.unigrams[PLACEHOLDER_SURNAME][0]
        
        # Placename placeholder
        if PLACEHOLDER_PLACENAME in self.unigrams:
            placeholders[PLACEHOLDER_PLACENAME] = self.unigrams[PLACEHOLDER_PLACENAME][0]
        
        # Forename placeholders by case
        for case in LATIN_CASES:
            placeholder = get_forename_placeholder(case)
            if placeholder in self.unigrams:
                placeholders[placeholder] = self.unigrams[placeholder][0]
        
        return placeholders
    
    def inject_names_from_placeholder(
        self,
        placeholder: str,
        names_with_frequencies: Dict[str, int],
        name_db: NameDatabase,
        top_contexts: int = 50,
        top_names: int = 1000
    ):
        """
        Replace a placeholder with actual names, distributing probability mass.
        
        This implements class-based LM expansion:
        1. Unigrams: ALL names get P(name) = P(placeholder) × (freq_name / total_freq)
        2. Bigrams/Trigrams: Only top-N most frequent names get n-gram expansion
           for top-K contexts. Rare names fall back to unigrams.
        
        Args:
            placeholder: The placeholder token to replace
            names_with_frequencies: Dict mapping name -> frequency
            name_db: The name database (unused but kept for API compatibility)
            top_contexts: Number of top contexts to expand (default 50)
            top_names: Number of top names to expand with n-grams (default 1000)
        """
        if placeholder not in self.unigrams:
            print(f"  Warning: Placeholder {placeholder} not found in unigrams")
            return
        
        placeholder_log_prob, placeholder_backoff = self.unigrams[placeholder]
        placeholder_prob = 10 ** placeholder_log_prob  # Convert from log10
        
        # Calculate total frequency for class-conditional probability
        total_freq = sum(names_with_frequencies.values())
        if total_freq == 0:
            print(f"  Warning: No frequencies for {placeholder}")
            return
        
        # Pre-compute log frequency ratios for all names
        log_freq_ratios = {}
        for name, freq in names_with_frequencies.items():
            if freq > 0:
                log_freq_ratios[name] = math.log10(freq / total_freq)
            else:
                log_freq_ratios[name] = -99.0
        
        # Get top-N most frequent names for n-gram expansion
        sorted_names = sorted(names_with_frequencies.items(), key=lambda x: x[1], reverse=True)
        top_names_for_ngrams = dict(sorted_names[:top_names])
        
        print(f"  {placeholder}:")
        print(f"    Unigram log prob: {placeholder_log_prob:.6f} (prob: {placeholder_prob:.6e})")
        print(f"    Total names: {len(names_with_frequencies):,}")
        print(f"    Names for n-gram expansion: {len(top_names_for_ngrams):,} (top by frequency)")
        
        # Remove the placeholder from unigrams
        del self.unigrams[placeholder]
        
        # Add each actual name with its share of the probability (unigrams for ALL names)
        added_unigrams = 0
        for name, freq in names_with_frequencies.items():
            # Skip if name already exists (don't overwrite)
            if name in self.unigrams:
                continue
            
            # Calculate this name's share of probability
            name_prob = placeholder_prob * (freq / total_freq)
            name_log_prob = math.log10(name_prob) if name_prob > 0 else -99.0
            
            # Use the same backoff weight as the placeholder
            self.unigrams[name] = (name_log_prob, placeholder_backoff)
            added_unigrams += 1
        
        print(f"    Added {added_unigrams:,} new unigrams (all names)")
        
        # Expand top-K bigram contexts (only for top-N names)
        added_bigrams = self._expand_bigram_contexts(
            placeholder, top_names_for_ngrams, log_freq_ratios, top_contexts
        )
        
        # Expand top-K trigram contexts (only for top-N names)
        added_trigrams = self._expand_trigram_contexts(
            placeholder, top_names_for_ngrams, log_freq_ratios, top_contexts
        )
        
        print(f"    Added {added_bigrams:,} new bigrams (top {top_names} names × top {top_contexts} contexts)")
        print(f"    Added {added_trigrams:,} new trigrams (top {top_names} names × top {top_contexts} contexts)")
        
        # Remove placeholder n-grams (they've been replaced by expanded versions)
        self._remove_placeholder_from_ngrams(placeholder)
        
        # Update counts
        self.ngram_counts[1] = len(self.unigrams)
        self.ngram_counts[2] = len(self.bigrams)
        self.ngram_counts[3] = len(self.trigrams)
    
    def _expand_bigram_contexts(
        self,
        placeholder: str,
        names_with_frequencies: Dict[str, int],
        log_freq_ratios: Dict[str, float],
        top_k: int
    ) -> int:
        """Expand top-K bigram contexts with all names."""
        contexts = self.placeholder_bigrams.get(placeholder, [])
        if not contexts:
            return 0
        
        # Sort by probability (descending) to get top-K most likely contexts
        sorted_contexts = sorted(contexts, key=lambda x: x[0], reverse=True)[:top_k]
        
        added = 0
        existing_bigrams = set((bg[1], bg[2]) for bg in self.bigrams)
        
        for log_prob, context_words, position, backoff in sorted_contexts:
            for name in names_with_frequencies:
                # Compute new probability: P(context with name) = P(context with placeholder) × P(name|class)
                new_log_prob = log_prob + log_freq_ratios[name]
                
                # Build the bigram based on placeholder position
                if position == 0:
                    # "PLACEHOLDER context" -> "name context"
                    w1, w2 = name, context_words[0]
                else:
                    # "context PLACEHOLDER" -> "context name"
                    w1, w2 = context_words[0], name
                
                # Skip if this bigram already exists
                if (w1, w2) in existing_bigrams:
                    continue
                
                self.bigrams.append((new_log_prob, w1, w2, backoff))
                existing_bigrams.add((w1, w2))
                added += 1
        
        return added
    
    def _expand_trigram_contexts(
        self,
        placeholder: str,
        names_with_frequencies: Dict[str, int],
        log_freq_ratios: Dict[str, float],
        top_k: int
    ) -> int:
        """Expand top-K trigram contexts with all names."""
        contexts = self.placeholder_trigrams.get(placeholder, [])
        if not contexts:
            return 0
        
        # Sort by probability (descending) to get top-K most likely contexts
        sorted_contexts = sorted(contexts, key=lambda x: x[0], reverse=True)[:top_k]
        
        added = 0
        existing_trigrams = set((tg[1], tg[2], tg[3]) for tg in self.trigrams)
        
        for log_prob, context_words, position in sorted_contexts:
            for name in names_with_frequencies:
                # Compute new probability: P(context with name) = P(context with placeholder) × P(name|class)
                new_log_prob = log_prob + log_freq_ratios[name]
                
                # Build the trigram based on placeholder position
                if position == 0:
                    # "PLACEHOLDER w2 w3" -> "name w2 w3"
                    w1, w2, w3 = name, context_words[0], context_words[1]
                elif position == 1:
                    # "w1 PLACEHOLDER w3" -> "w1 name w3"
                    w1, w2, w3 = context_words[0], name, context_words[1]
                else:
                    # "w1 w2 PLACEHOLDER" -> "w1 w2 name"
                    w1, w2, w3 = context_words[0], context_words[1], name
                
                # Skip if this trigram already exists
                if (w1, w2, w3) in existing_trigrams:
                    continue
                
                self.trigrams.append((new_log_prob, w1, w2, w3))
                existing_trigrams.add((w1, w2, w3))
                added += 1
        
        return added
    
    def _remove_placeholder_from_ngrams(self, placeholder: str):
        """Remove n-grams containing a placeholder token."""
        # Filter bigrams
        original_bigram_count = len(self.bigrams)
        self.bigrams = [
            bg for bg in self.bigrams
            if bg[1] != placeholder and bg[2] != placeholder
        ]
        removed_bigrams = original_bigram_count - len(self.bigrams)
        
        # Filter trigrams
        original_trigram_count = len(self.trigrams)
        self.trigrams = [
            tg for tg in self.trigrams
            if tg[1] != placeholder and tg[2] != placeholder and tg[3] != placeholder
        ]
        removed_trigrams = original_trigram_count - len(self.trigrams)
        
        if removed_bigrams > 0 or removed_trigrams > 0:
            print(f"    Removed {removed_bigrams} bigrams, {removed_trigrams} trigrams")
        
        # Update counts
        self.ngram_counts[2] = len(self.bigrams)
        self.ngram_counts[3] = len(self.trigrams)
    
    def write(self, output_path: Path):
        """Write the modified ARPA file."""
        print(f"Writing modified ARPA to: {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Write header
            f.write('\\data\\\n')
            for order in sorted(self.ngram_counts.keys()):
                f.write(f'ngram {order}={self.ngram_counts[order]}\n')
            f.write('\n')
            
            # Write unigrams
            f.write('\\1-grams:\n')
            for word, (log_prob, backoff) in sorted(self.unigrams.items()):
                f.write(f'{log_prob}\t{word}\t{backoff}\n')
            f.write('\n')
            
            # Write bigrams
            f.write('\\2-grams:\n')
            for log_prob, w1, w2, backoff in self.bigrams:
                f.write(f'{log_prob}\t{w1} {w2}\t{backoff}\n')
            f.write('\n')
            
            # Write trigrams
            f.write('\\3-grams:\n')
            for log_prob, w1, w2, w3 in self.trigrams:
                f.write(f'{log_prob}\t{w1} {w2} {w3}\n')
            f.write('\n')
            
            # Write any higher-order n-grams
            for order in sorted(self.higher_ngrams.keys()):
                f.write(f'\\{order}-grams:\n')
                for line in self.higher_ngrams[order]:
                    f.write(line + '\n')
                f.write('\n')
            
            f.write('\\end\\\n')
        
        print(f"✓ Modified ARPA written")


def compile_arpa_to_binary(arpa_path: Path, binary_path: Path) -> bool:
    """Compile ARPA to binary format using build_binary."""
    build_binary_path = find_kenlm_binary('build_binary')
    
    if not build_binary_path:
        print("Warning: build_binary not found. Binary compilation skipped.")
        return False
    
    try:
        cmd = [build_binary_path, str(arpa_path), str(binary_path)]
        print(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600
        )
        
        if result.returncode != 0:
            print(f"Error building binary model:")
            print(result.stderr)
            return False
        
        print(f"✓ Binary model saved to: {binary_path}")
        return True
        
    except Exception as e:
        print(f"Error during binary compilation: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Train KenLM with placeholder-based name injection'
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        default=None,
        help='Base directory containing step2a_merged.json files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='kenlm_model',
        help='Output directory for model files'
    )
    parser.add_argument(
        '--db',
        type=str,
        default='cp40_records.db',
        help='Path to database file'
    )
    parser.add_argument(
        '--order',
        type=int,
        default=3,
        help='N-gram order (default: 3)'
    )
    parser.add_argument(
        '--memory',
        type=str,
        default='80%',
        help='Memory limit for training'
    )
    parser.add_argument(
        '--min-words',
        type=int,
        default=2,
        help='Minimum words per text'
    )
    parser.add_argument(
        '--keep-intermediate',
        action='store_true',
        help='Keep intermediate files (placeholder corpus, initial ARPA)'
    )
    parser.add_argument(
        '--top-contexts',
        type=int,
        default=50,
        help='Number of top bigram/trigram contexts to expand for each placeholder (default: 50)'
    )
    parser.add_argument(
        '--top-names',
        type=int,
        default=1000,
        help='Number of top (most frequent) names to expand with n-gram contexts. '
             'All names still get unigrams for backoff. (default: 1000)'
    )
    
    args = parser.parse_args()
    
    # Determine paths
    script_dir = Path(__file__).parent
    
    if args.base_dir:
        base_dir = Path(args.base_dir)
    else:
        base_dir = script_dir / "cp40_processing" / "output"
    
    if not base_dir.exists():
        print(f"Error: Directory {base_dir} does not exist")
        return 1
    
    db_path = script_dir / args.db
    if not db_path.exists():
        print(f"Error: Database not found: {db_path}")
        return 1
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load name database
    print("\n" + "="*60)
    print("Step 1: Loading name database")
    print("="*60)
    name_db = NameDatabase(db_path)
    
    # Step 2: Find and extract corpus texts
    print("\n" + "="*60)
    print("Step 2: Extracting corpus texts")
    print("="*60)
    file_paths = find_all_step2a_files(base_dir)
    if not file_paths:
        print("No step2a_merged.json files found")
        return 1
    
    texts = extract_merged_texts(file_paths, min_words=args.min_words)
    if not texts:
        print("No texts extracted")
        return 1
    
    # Step 3: Substitute placeholders
    print("\n" + "="*60)
    print("Step 3: Substituting names with placeholders")
    print("="*60)
    placeholder_texts, sub_counts = substitute_placeholders(texts, name_db)
    
    # Step 4: Train initial KenLM model
    print("\n" + "="*60)
    print("Step 4: Training initial KenLM model")
    print("="*60)
    
    placeholder_corpus_file = output_dir / "placeholder_corpus.txt"
    prepare_training_text(placeholder_texts, placeholder_corpus_file)
    
    initial_arpa = output_dir / f"initial_{args.order}gram.arpa"
    success = train_kenlm_model(
        placeholder_corpus_file,
        initial_arpa,
        order=args.order,
        memory=args.memory
    )
    
    if not success:
        print("Failed to train initial model")
        return 1
    
    # Step 5: Parse ARPA and inject names
    print("\n" + "="*60)
    print("Step 5: Parsing ARPA and injecting actual names")
    print("="*60)
    
    arpa = ArpaParser(initial_arpa)
    arpa.parse()
    
    # Get placeholder probabilities
    placeholder_probs = arpa.get_placeholder_probabilities()
    print("\nPlaceholder probabilities found:")
    for placeholder, log_prob in placeholder_probs.items():
        print(f"  {placeholder}: {log_prob:.6f}")
    
    # Inject actual names for each placeholder
    print(f"\nInjecting actual names...")
    print(f"  Top contexts per placeholder: {args.top_contexts}")
    print(f"  Top names for n-gram expansion: {args.top_names}")
    print(f"  (All names get unigrams; only top names get bigram/trigram expansion)")
    
    # Inject surnames
    if name_db.surname_frequencies:
        arpa.inject_names_from_placeholder(
            PLACEHOLDER_SURNAME,
            name_db.surname_frequencies,
            name_db,
            top_contexts=args.top_contexts,
            top_names=args.top_names
        )
    
    # Inject placenames
    if name_db.placename_frequencies:
        arpa.inject_names_from_placeholder(
            PLACEHOLDER_PLACENAME,
            name_db.placename_frequencies,
            name_db,
            top_contexts=args.top_contexts,
            top_names=args.top_names
        )
    
    # Inject forenames by case
    for case in LATIN_CASES:
        placeholder = get_forename_placeholder(case)
        if name_db.forename_case_frequencies[case]:
            arpa.inject_names_from_placeholder(
                placeholder,
                name_db.forename_case_frequencies[case],
                name_db,
                top_contexts=args.top_contexts,
                top_names=args.top_names
            )
    
    # Step 6: Write modified ARPA
    print("\n" + "="*60)
    print("Step 6: Writing modified ARPA file")
    print("="*60)
    
    final_arpa = output_dir / f"kenlm_model_{args.order}gram.arpa"
    arpa.write(final_arpa)
    
    # Step 7: Compile to binary
    print("\n" + "="*60)
    print("Step 7: Compiling to binary format")
    print("="*60)
    
    final_binary = output_dir / f"kenlm_model_{args.order}gram.klm"
    compile_arpa_to_binary(final_arpa, final_binary)
    
    # Cleanup intermediate files if requested
    if not args.keep_intermediate:
        print("\nCleaning up intermediate files...")
        if placeholder_corpus_file.exists():
            placeholder_corpus_file.unlink()
        if initial_arpa.exists():
            initial_arpa.unlink()
    
    # Close database
    name_db.close()
    
    # Summary
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nOutput files:")
    if final_arpa.exists():
        size_mb = final_arpa.stat().st_size / (1024 * 1024)
        print(f"  ARPA: {final_arpa} ({size_mb:.2f} MB)")
    if final_binary.exists():
        size_mb = final_binary.stat().st_size / (1024 * 1024)
        print(f"  Binary: {final_binary} ({size_mb:.2f} MB)")
    
    print(f"\nModel statistics:")
    print(f"  - N-gram order: {args.order}")
    print(f"  - Original sentences: {len(texts):,}")
    print(f"  - Top contexts expanded: {args.top_contexts}")
    print(f"  - Top names for n-gram expansion: {args.top_names}")
    print(f"  - Final unigrams: {arpa.ngram_counts.get(1, 0):,}")
    print(f"  - Final bigrams: {arpa.ngram_counts.get(2, 0):,}")
    print(f"  - Final trigrams: {arpa.ngram_counts.get(3, 0):,}")
    
    print("\nHow it works (class-based LM approach):")
    print("  1. ALL names get unigram probabilities (for backoff)")
    print("  ")
    print(f"  2. Top {args.top_names} most frequent names get n-gram expansion:")
    print("     For common contexts like 'de PLACEHOLDER_SURNAME':")
    print("       - Expanded to 'de Smith', 'de Jones', etc.")
    print("       - Each with probability: P(de PLACEHOLDER) × (freq_name / total)")
    print("  ")
    print(f"  3. Rare names (not in top {args.top_names}) and rare contexts:")
    print("     - Fall back to unigram P(name)")
    print("  ")
    print("  This is mathematically equivalent to:")
    print("    P(word|context) = P(class|context) × P(word|class)")
    
    return 0


if __name__ == "__main__":
    exit(main())

