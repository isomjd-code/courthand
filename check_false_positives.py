#!/usr/bin/env python3
"""Check for potential false positive name matches in the database."""

import sqlite3
from pathlib import Path
from collections import defaultdict

# Common Latin words that should NOT be names
COMMON_LATIN_WORDS = {
    # Days/Time
    'die', 'dies', 'diem', 'anno', 'annum', 'anni', 'regni', 'regnum',
    'lune', 'martis', 'mercurii', 'jovis', 'veneris', 'sabbati', 'dominica',
    'prima', 'secunda', 'tertia', 'quarta', 'quinta', 'sexta', 'septima',
    'octava', 'nona', 'decima', 'primo', 'secundo', 'tertio',
    # Legal terms
    'lege', 'legem', 'iure', 'iuris', 'curia', 'curiam', 'placito', 'placitum',
    'breve', 'brevis', 'carta', 'cartam', 'terra', 'terram', 'terre',
    'villa', 'villam', 'ville', 'comitatu', 'comitatus', 'hundredo',
    'parochia', 'parochiam', 'ecclesia', 'ecclesiam', 'capella',
    # Common prepositions/conjunctions
    'de', 'in', 'ad', 'per', 'pro', 'cum', 'sine', 'super', 'sub', 'ante', 'post',
    'et', 'vel', 'aut', 'nec', 'neque', 'sed', 'quod', 'quia', 'quam',
    # Pronouns/articles
    'qui', 'que', 'quod', 'ipse', 'ipsa', 'ipsum', 'ipsius', 'eius', 'eorum',
    'hic', 'hec', 'hoc', 'huius', 'ille', 'illa', 'illud', 'illius',
    # Common verbs
    'est', 'sunt', 'fuit', 'fuerunt', 'esse', 'fore', 'habet', 'habuit',
    'dicit', 'dicunt', 'dixit', 'venit', 'venerunt', 'fecit', 'fecerunt',
    # Common nouns
    'rex', 'regis', 'rege', 'regina', 'regine', 'dominus', 'domini', 'domino',
    'filius', 'filii', 'filio', 'filium', 'filia', 'filie', 'uxor', 'uxoris',
    'homo', 'hominis', 'homine', 'hominem', 'femina', 'femine',
    'pater', 'patris', 'patre', 'mater', 'matris', 'matre',
    # Religious
    'sancti', 'sancte', 'sancto', 'sanctum', 'beati', 'beate', 'beato',
    'marie', 'maria', 'mariam', 'michaelis', 'michael', 'johannis', 'johannes',
    'petri', 'petrus', 'pauli', 'paulus', 'trinitatis', 'trinitate',
    # Numbers
    'unus', 'una', 'unum', 'duo', 'duos', 'duas', 'tres', 'tria',
    'quatuor', 'quinque', 'sex', 'septem', 'octo', 'novem', 'decem',
    # Common adjectives
    'maior', 'maioris', 'minor', 'minoris', 'bonus', 'bona', 'bonum',
    'magnus', 'magna', 'magnum', 'parvus', 'parva', 'parvum',
    'novus', 'nova', 'novum', 'vetus', 'veteris', 'liber', 'libera', 'liberum',
    # Other common terms
    'modo', 'tunc', 'nunc', 'inde', 'unde', 'ibidem', 'item', 'videlicet',
    'scilicet', 'sicut', 'prout', 'tanquam', 'quasi',
}

# Very short words (2 chars or less) are suspicious
MIN_SAFE_LENGTH = 3

def check_database(db_path: Path):
    """Check database for potential false positive matches."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    results = {
        'surnames': {'common_latin': [], 'too_short': [], 'single_char': []},
        'forenames': {'common_latin': [], 'too_short': [], 'single_char': []},
        'placenames': {'common_latin': [], 'too_short': [], 'single_char': []},
    }
    
    # Check surnames
    print("\n" + "="*60)
    print("Checking SURNAMES")
    print("="*60)
    try:
        cursor = conn.execute("SELECT DISTINCT surname FROM surnames WHERE surname IS NOT NULL")
        for row in cursor:
            word = row['surname']
            if not word:
                continue
            word_lower = word.lower()
            
            if len(word) == 1:
                results['surnames']['single_char'].append(word)
            elif len(word) < MIN_SAFE_LENGTH:
                results['surnames']['too_short'].append(word)
            
            if word_lower in COMMON_LATIN_WORDS:
                results['surnames']['common_latin'].append(word)
    except sqlite3.OperationalError as e:
        print(f"  Error: {e}")
    
    # Check forenames (latin forms)
    print("\n" + "="*60)
    print("Checking FORENAMES (Latin forms)")
    print("="*60)
    try:
        cursor = conn.execute("""
            SELECT DISTINCT latin_abbreviated, case_name 
            FROM forename_latin_forms 
            WHERE latin_abbreviated IS NOT NULL
        """)
        for row in cursor:
            word = row['latin_abbreviated']
            case_name = row['case_name']
            if not word:
                continue
            word_lower = word.lower()
            
            if len(word) == 1:
                results['forenames']['single_char'].append(f"{word} ({case_name})")
            elif len(word) < MIN_SAFE_LENGTH:
                results['forenames']['too_short'].append(f"{word} ({case_name})")
            
            if word_lower in COMMON_LATIN_WORDS:
                results['forenames']['common_latin'].append(f"{word} ({case_name})")
    except sqlite3.OperationalError as e:
        print(f"  Error: {e}")
    
    # Check placenames
    print("\n" + "="*60)
    print("Checking PLACENAMES")
    print("="*60)
    try:
        cursor = conn.execute("SELECT DISTINCT name FROM places WHERE name IS NOT NULL")
        for row in cursor:
            word = row['name']
            if not word:
                continue
            word_lower = word.lower()
            
            if len(word) == 1:
                results['placenames']['single_char'].append(word)
            elif len(word) < MIN_SAFE_LENGTH:
                results['placenames']['too_short'].append(word)
            
            if word_lower in COMMON_LATIN_WORDS:
                results['placenames']['common_latin'].append(word)
    except sqlite3.OperationalError as e:
        print(f"  Error: {e}")
    
    # Also check place_latin_forms
    try:
        cursor = conn.execute("""
            SELECT DISTINCT latin_abbreviated 
            FROM place_latin_forms 
            WHERE latin_abbreviated IS NOT NULL
        """)
        for row in cursor:
            word = row['latin_abbreviated']
            if not word:
                continue
            word_lower = word.lower()
            
            if len(word) == 1:
                if word not in results['placenames']['single_char']:
                    results['placenames']['single_char'].append(word)
            elif len(word) < MIN_SAFE_LENGTH:
                if word not in results['placenames']['too_short']:
                    results['placenames']['too_short'].append(word)
            
            if word_lower in COMMON_LATIN_WORDS:
                if word not in results['placenames']['common_latin']:
                    results['placenames']['common_latin'].append(word)
    except sqlite3.OperationalError:
        pass
    
    conn.close()
    
    # Print results
    print("\n" + "="*60)
    print("POTENTIAL FALSE POSITIVES FOUND")
    print("="*60)
    
    for category, issues in results.items():
        print(f"\n{category.upper()}:")
        
        if issues['single_char']:
            print(f"  Single character ({len(issues['single_char'])}): {issues['single_char'][:20]}")
            if len(issues['single_char']) > 20:
                print(f"    ... and {len(issues['single_char']) - 20} more")
        
        if issues['too_short']:
            print(f"  Too short (<{MIN_SAFE_LENGTH} chars) ({len(issues['too_short'])}): {issues['too_short'][:20]}")
            if len(issues['too_short']) > 20:
                print(f"    ... and {len(issues['too_short']) - 20} more")
        
        if issues['common_latin']:
            print(f"  Common Latin words ({len(issues['common_latin'])}): {issues['common_latin'][:30]}")
            if len(issues['common_latin']) > 30:
                print(f"    ... and {len(issues['common_latin']) - 30} more")
    
    return results


if __name__ == "__main__":
    # Try both database paths
    db_paths = [
        Path("cp40_database_new.sqlite"),
        Path("cp40_records.db"),
    ]
    
    for db_path in db_paths:
        if db_path.exists():
            print(f"\nChecking database: {db_path}")
            results = check_database(db_path)
            break
    else:
        print("No database found!")

