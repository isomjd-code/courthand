import os
import json
import argparse
import sys
import difflib
from collections import defaultdict
from dataclasses import dataclass

# --- Data Structures for Aggregation ---

@dataclass
class MetricStats:
    total_count: int = 0
    exact_matches: int = 0
    sum_similarity: float = 0.0

    @property
    def accuracy(self):
        return (self.exact_matches / self.total_count * 100) if self.total_count > 0 else 0.0

    @property
    def avg_similarity(self):
        return (self.sum_similarity / self.total_count * 100) if self.total_count > 0 else 0.0

# --- Core Logic ---

def load_record(filepath):
    """Safely loads a JSON record."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[Error] Could not read {filepath}: {e}", file=sys.stderr)
        return None

def update_confusion_matrix(matrix, gt_str, ai_str):
    """
    Aligns two strings and updates the confusion matrix.
    Tracks: Matches (Equal), Substitutions (Replace), and Deletions (Delete).
    """
    # Normalize inputs
    gt_str = str(gt_str) if gt_str else ""
    ai_str = str(ai_str) if ai_str else ""
    
    # Use SequenceMatcher to find alignment
    matcher = difflib.SequenceMatcher(None, gt_str, ai_str)
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            # Correct transcription
            for k in range(i2 - i1):
                char = gt_str[i1+k]
                matrix[char][char] += 1
                
        elif tag == 'replace':
            # Substitution
            sub_gt = gt_str[i1:i2]
            sub_ai = ai_str[j1:j2]
            
            # Map 1-to-1 as far as possible
            min_len = min(len(sub_gt), len(sub_ai))
            for k in range(min_len):
                matrix[sub_gt[k]][sub_ai[k]] += 1
                
            # If GT segment was longer, the extras were effectively deleted (missed)
            if len(sub_gt) > len(sub_ai):
                for k in range(min_len, len(sub_gt)):
                    matrix[sub_gt[k]]['[MISS]'] += 1

        elif tag == 'delete':
            # GT has characters, AI missed them completely
            for k in range(i1, i2):
                matrix[gt_str[k]]['[MISS]'] += 1

        # We ignore 'insert' because we calculate percentages based on GT characters.

def process_directory(base_path):
    """
    Traverses directories to find master_record.json files and aggregates statistics.
    """
    
    # Aggregators
    global_stats = MetricStats()
    category_stats = defaultdict(MetricStats)
    field_stats = defaultdict(MetricStats)
    # Track subcategory (field_name) stats within each category
    subcategory_stats = defaultdict(lambda: defaultdict(MetricStats))
    
    # Confusion Matrix: matrix[GroundTruthChar][AIChar] = Count
    char_confusion_matrix = defaultdict(lambda: defaultdict(int))
    
    # To track lowest performing specific fields for the "Need Attention" section
    low_performing_instances = []

    files_processed = 0
    files_with_reports = 0

    print(f"Scanning directory: {base_path} ...\n")

    for root, dirs, files in os.walk(base_path):
        if 'master_record.json' in files:
            file_path = os.path.join(root, 'master_record.json')
            data = load_record(file_path)
            
            if not data:
                continue
                
            files_processed += 1
            
            # Navigate to the validation report section
            val_report = data.get('validation_report', {}).get('metrics', {})
            
            if not val_report:
                continue
                
            files_with_reports += 1
            
            # We use the 'field_comparisons' array to rebuild global stats
            comparisons = val_report.get('field_comparisons', [])
            
            for comp in comparisons:
                cat = comp.get('category', 'Uncategorized')
                field_name = comp.get('field_name', 'Unknown Field')
                is_match = comp.get('is_match', False)
                sim_score = comp.get('similarity_score', 0.0)
                gt_val = comp.get('gt_value')
                ai_val = comp.get('ai_value')
                
                # Update Global
                global_stats.total_count += 1
                if is_match: global_stats.exact_matches += 1
                global_stats.sum_similarity += sim_score
                
                # Update Category
                category_stats[cat].total_count += 1
                if is_match: category_stats[cat].exact_matches += 1
                category_stats[cat].sum_similarity += sim_score
                
                # Update Subcategory (field_name) within Category
                subcategory_stats[cat][field_name].total_count += 1
                if is_match: subcategory_stats[cat][field_name].exact_matches += 1
                subcategory_stats[cat][field_name].sum_similarity += sim_score
                
                # Update Specific Field
                field_stats[field_name].total_count += 1
                if is_match: field_stats[field_name].exact_matches += 1
                field_stats[field_name].sum_similarity += sim_score

                # --- Update Confusion Matrix ---
                # We analyze ALL Agent Names (Matches AND Mismatches) to get the baseline stats
                if cat == 'Agents' and 'Name' in field_name:
                    update_confusion_matrix(char_confusion_matrix, gt_val, ai_val)

                # Track specific failures for the report (if similarity < 0.85)
                if sim_score < 0.85:
                    low_performing_instances.append({
                        'file': os.path.basename(root),
                        'field': field_name,
                        'score': sim_score,
                        'gt': str(gt_val)[:50],
                        'ai': str(ai_val)[:50]
                    })

    return {
        'files_processed': files_processed,
        'files_with_reports': files_with_reports,
        'global': global_stats,
        'by_category': category_stats,
        'by_subcategory': subcategory_stats,
        'by_field': field_stats,
        'confusion_matrix': char_confusion_matrix,
        'issues': low_performing_instances
    }

# --- Reporting ---

def print_separator(char='-', length=100):
    print(char * length)

def generate_report(data):
    if data['files_processed'] == 0:
        print("No 'master_record.json' files found in the specified directory.")
        return

    print_separator('=')
    print(f"COMPREHENSIVE VALIDATION REPORT")
    print_separator('=')
    print(f"Total Files Scanned:      {data['files_processed']}")
    print(f"Total Data Fields Checked: {data['global'].total_count}")
    print_separator()
    
    # 1. Executive Summary
    print(f"GLOBAL PERFORMANCE METRICS")
    print(f"Overall Exact Match Accuracy:   {data['global'].accuracy:.2f}%")
    print(f"Overall Semantic Similarity:    {data['global'].avg_similarity:.2f}%")
    print_separator()

    # 2. Performance by Category (with Subcategory breakdown)
    print(f"CATEGORY STATISTICS")
    print_separator('-', 100)
    
    sorted_cats = sorted(data['by_category'].items(), key=lambda x: x[1].avg_similarity, reverse=True)
    
    for cat, stats in sorted_cats:
        # Print category header
        print(f"\n{cat}")
        print(f"{'  SUBCATEGORY':<30} | {'COUNT':<8} | {'ACCURACY':<10} | {'AVG SIMILARITY':<15}")
        print_separator('-', 75)
        
        # Get subcategories for this category
        subcats = data['by_subcategory'].get(cat, {})
        if subcats:
            # Sort subcategories by average similarity (descending)
            sorted_subcats = sorted(subcats.items(), key=lambda x: x[1].avg_similarity, reverse=True)
            for subcat, subcat_stats in sorted_subcats:
                print(f"  {subcat:<28} | {subcat_stats.total_count:<8} | {subcat_stats.accuracy:>6.2f}%   | {subcat_stats.avg_similarity:>10.2f}%")
        else:
            print(f"  {'(No subcategories)':<28} | {'':<8} | {'':<10} | {'':<15}")
        
        # Print category summary
        print(f"{'  [CATEGORY TOTAL]':<28} | {stats.total_count:<8} | {stats.accuracy:>6.2f}%   | {stats.avg_similarity:>10.2f}%")
    
    print_separator()

    # 3. Character Transcription Analysis
    print(f"CHARACTER TRANSCRIPTION PROFILE (Agent Names Only)")
    print(f"Shows the accuracy for each Ground Truth character.")
    print(f"Sorted by ASCENDING Accuracy (Worst performance at the top).")
    print_separator('-', 100)
    print(f"{'GT CHAR':<8} | {'TOTAL':<6} | {'ACCURACY':<8} | {'TRANSCRIPTION BREAKDOWN'}")
    print_separator('-', 100)
    
    matrix = data['confusion_matrix']
    
    char_stats = []
    
    for gt_char, confusion_dict in matrix.items():
        total_occurrences = sum(confusion_dict.values())
        correct_count = confusion_dict.get(gt_char, 0)
        
        # Calculate accuracy
        accuracy = (correct_count / total_occurrences) * 100 if total_occurrences > 0 else 0.0
        
        # Calculate breakdown percentages
        breakdown = []
        for ai_char, count in confusion_dict.items():
            if count > 0:
                pct = (count / total_occurrences) * 100
                breakdown.append((ai_char, pct))
        
        # Sort breakdown: Correct first, then by percentage descending
        breakdown.sort(key=lambda x: (x[0] == gt_char, x[1]), reverse=True)
        
        char_stats.append({
            'char': gt_char,
            'total': total_occurrences,
            'accuracy': accuracy,
            'breakdown': breakdown
        })

    # Sort primarily by Accuracy (Ascending - low first), secondary by Total Count (Descending - high freq first)
    char_stats.sort(key=lambda x: (x['accuracy'], -x['total']))
    
    for stat in char_stats:
        # Optional: Hide chars that appeared very rarely AND were perfect (100% acc) to reduce noise.
        # But we show low accuracy even if rare (e.g. 0%).
        if stat['total'] < 3 and stat['accuracy'] == 100.0:
            continue

        gt_disp = f"'{stat['char']}'" if stat['char'].strip() else "[SPC]"
        
        # Format breakdown string
        breakdown_str_parts = []
        for ai_char, pct in stat['breakdown']:
            ai_disp = f"'{ai_char}'" if ai_char.strip() else "[SPC]"
            if ai_char == stat['char']:
                # Label the correct one
                breakdown_str_parts.append(f"{ai_disp} (OK): {pct:.1f}%")
            else:
                breakdown_str_parts.append(f"{ai_disp}: {pct:.1f}%")
        
        breakdown_str = ", ".join(breakdown_str_parts)
        if len(breakdown_str) > 60:
            breakdown_str = breakdown_str[:57] + "..."

        print(f"{gt_disp:<8} | {stat['total']:<6} | {stat['accuracy']:5.1f}%   | {breakdown_str}")

    print_separator()

    # 4. Low Performance Instances
    issues = data['issues']
    if issues:
        print(f"LOWEST PERFORMING INSTANCES (Similarity < 85%) - Top 15 shown")
        print(f"{'CASE ID':<20} | {'FIELD':<20} | {'SCORE':<6} | {'GROUND TRUTH vs AI'}")
        print_separator('-', 100)
        
        issues.sort(key=lambda x: x['score'])
        
        for issue in issues[:15]:
            print(f"{issue['file']:<20} | {issue['field']:<20} | {issue['score']*100:>5.1f}% | GT: {issue['gt']}")
            print(f"{'':<20} | {'':<20} | {'':<6} | AI: {issue['ai']}")
            print("-" * 100)
    else:
        print("No fields fell below the 85% similarity threshold.")
    
    print_separator('=')

# --- Entry Point ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate aggregate validation report from master_record.json files.")
    parser.add_argument("directory", nargs='?', default="cp40_processing/output", help="Root directory to search (default: cp40_processing/output)")
    
    args = parser.parse_args()
    
    # Expand user home directory if path starts with ~
    search_path = os.path.expanduser(args.directory)
    
    # Convert to absolute path if relative
    if not os.path.isabs(search_path):
        search_path = os.path.abspath(search_path)
    
    if not os.path.isdir(search_path):
        print(f"Error: Directory '{search_path}' does not exist.")
        sys.exit(1)
        
    results = process_directory(search_path)
    generate_report(results)