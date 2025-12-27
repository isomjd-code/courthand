#!/usr/bin/env python3
"""
Statistical Analysis of Field Comparisons from Master Records

This script parses all master_record.json files and generates detailed
statistical summaries of field_comparisons by category and field name.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Optional
import statistics


class FieldComparisonStats:
    """Statistics for a single field or category."""
    
    def __init__(self):
        self.total_count = 0
        self.matches = 0
        self.non_matches = 0
        self.similarity_scores = []
        self.similarity_basis_counts = defaultdict(int)
        self.cases_processed = set()  # Track which cases contributed
        
    @property
    def accuracy(self) -> float:
        """Calculate accuracy percentage."""
        if self.total_count == 0:
            return 0.0
        return (self.matches / self.total_count) * 100.0
    
    @property
    def avg_similarity(self) -> float:
        """Calculate average similarity score."""
        if not self.similarity_scores:
            return 0.0
        return statistics.mean(self.similarity_scores)
    
    @property
    def median_similarity(self) -> float:
        """Calculate median similarity score."""
        if not self.similarity_scores:
            return 0.0
        return statistics.median(self.similarity_scores)
    
    @property
    def min_similarity(self) -> float:
        """Get minimum similarity score."""
        if not self.similarity_scores:
            return 0.0
        return min(self.similarity_scores)
    
    @property
    def max_similarity(self) -> float:
        """Get maximum similarity score."""
        if not self.similarity_scores:
            return 0.0
        return max(self.similarity_scores)
    
    @property
    def std_dev_similarity(self) -> float:
        """Calculate standard deviation of similarity scores."""
        if len(self.similarity_scores) < 2:
            return 0.0
        return statistics.stdev(self.similarity_scores)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for output."""
        return {
            "total_count": self.total_count,
            "matches": self.matches,
            "non_matches": self.non_matches,
            "accuracy_percent": round(self.accuracy, 2),
            "avg_similarity": round(self.avg_similarity, 4),
            "median_similarity": round(self.median_similarity, 4),
            "min_similarity": round(self.min_similarity, 4),
            "max_similarity": round(self.max_similarity, 4),
            "std_dev_similarity": round(self.std_dev_similarity, 4),
            "similarity_basis_distribution": dict(self.similarity_basis_counts),
            "unique_cases": len(self.cases_processed)
        }


class FieldComparisonAnalyzer:
    """Analyzer for field comparisons across all master records."""
    
    def __init__(self, base_path: str = "cp40_processing/output"):
        self.base_path = Path(base_path)
        self.category_stats: Dict[str, FieldComparisonStats] = defaultdict(FieldComparisonStats)
        self.field_stats: Dict[str, FieldComparisonStats] = defaultdict(FieldComparisonStats)
        self.category_field_stats: Dict[str, Dict[str, FieldComparisonStats]] = defaultdict(lambda: defaultdict(FieldComparisonStats))
        self.global_stats = FieldComparisonStats()
        self.files_processed = 0
        self.files_with_comparisons = 0
        self.files_skipped = 0
        self.errors = []
        
    def find_master_records(self) -> List[Path]:
        """Find all master_record.json files."""
        master_records = []
        if not self.base_path.exists():
            print(f"Warning: Base path {self.base_path} does not exist.")
            return master_records
            
        for root, dirs, files in os.walk(self.base_path):
            if 'master_record.json' in files:
                master_records.append(Path(root) / 'master_record.json')
        
        return sorted(master_records)
    
    def load_master_record(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Load and parse a master_record.json file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.errors.append(f"Error loading {file_path}: {e}")
            return None
    
    def extract_field_comparisons(self, data: Dict[str, Any], case_id: str) -> List[Dict[str, Any]]:
        """Extract field_comparisons from master record data."""
        try:
            validation_report = data.get('validation_report', {})
            metrics = validation_report.get('metrics', {})
            comparisons = metrics.get('field_comparisons', [])
            return comparisons
        except Exception as e:
            self.errors.append(f"Error extracting comparisons from {case_id}: {e}")
            return []
    
    def process_comparison(self, comp: Dict[str, Any], case_id: str):
        """Process a single field comparison and update statistics."""
        field_name = comp.get('field_name', 'Unknown Field')
        category = comp.get('category', 'Uncategorized')
        is_match = comp.get('is_match', False)
        similarity_score = comp.get('similarity_score', 0.0)
        similarity_basis = comp.get('similarity_basis', 'unknown')
        
        # Update global stats
        self.global_stats.total_count += 1
        if is_match:
            self.global_stats.matches += 1
        else:
            self.global_stats.non_matches += 1
        self.global_stats.similarity_scores.append(similarity_score)
        self.global_stats.similarity_basis_counts[similarity_basis] += 1
        self.global_stats.cases_processed.add(case_id)
        
        # Update category stats
        cat_stats = self.category_stats[category]
        cat_stats.total_count += 1
        if is_match:
            cat_stats.matches += 1
        else:
            cat_stats.non_matches += 1
        cat_stats.similarity_scores.append(similarity_score)
        cat_stats.similarity_basis_counts[similarity_basis] += 1
        cat_stats.cases_processed.add(case_id)
        
        # Update field stats (across all categories)
        field_stats = self.field_stats[field_name]
        field_stats.total_count += 1
        if is_match:
            field_stats.matches += 1
        else:
            field_stats.non_matches += 1
        field_stats.similarity_scores.append(similarity_score)
        field_stats.similarity_basis_counts[similarity_basis] += 1
        field_stats.cases_processed.add(case_id)
        
        # Update category-field stats (field within specific category)
        cat_field_stats = self.category_field_stats[category][field_name]
        cat_field_stats.total_count += 1
        if is_match:
            cat_field_stats.matches += 1
        else:
            cat_field_stats.non_matches += 1
        cat_field_stats.similarity_scores.append(similarity_score)
        cat_field_stats.similarity_basis_counts[similarity_basis] += 1
        cat_field_stats.cases_processed.add(case_id)
    
    def analyze(self):
        """Analyze all master records."""
        print(f"Searching for master_record.json files in {self.base_path}...")
        master_records = self.find_master_records()
        print(f"Found {len(master_records)} master_record.json files.\n")
        
        for file_path in master_records:
            self.files_processed += 1
            case_id = file_path.parent.name
            
            data = self.load_master_record(file_path)
            if not data:
                self.files_skipped += 1
                continue
            
            comparisons = self.extract_field_comparisons(data, case_id)
            if not comparisons:
                self.files_skipped += 1
                continue
            
            self.files_with_comparisons += 1
            for comp in comparisons:
                self.process_comparison(comp, case_id)
            
            if self.files_processed % 10 == 0:
                print(f"Processed {self.files_processed} files...")
        
        print(f"\nAnalysis complete!")
        print(f"  Files processed: {self.files_processed}")
        print(f"  Files with comparisons: {self.files_with_comparisons}")
        print(f"  Files skipped: {self.files_skipped}")
        if self.errors:
            print(f"  Errors encountered: {len(self.errors)}")
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive statistical report."""
        report = {
            "summary": {
                "files_processed": self.files_processed,
                "files_with_comparisons": self.files_with_comparisons,
                "files_skipped": self.files_skipped,
                "errors": len(self.errors),
                "global_statistics": self.global_stats.to_dict()
            },
            "by_category": {
                cat: stats.to_dict()
                for cat, stats in sorted(self.category_stats.items())
            },
            "by_field_name": {
                field: stats.to_dict()
                for field, stats in sorted(self.field_stats.items())
            },
            "by_category_and_field": {
                cat: {
                    field: stats.to_dict()
                    for field, stats in sorted(field_stats.items())
                }
                for cat, field_stats in sorted(self.category_field_stats.items())
            }
        }
        
        return report
    
    def print_summary(self):
        """Print a human-readable summary to console."""
        print("\n" + "="*80)
        print("FIELD COMPARISON STATISTICAL SUMMARY")
        print("="*80)
        
        print(f"\n📊 GLOBAL STATISTICS")
        print(f"  Total Comparisons: {self.global_stats.total_count:,}")
        print(f"  Matches: {self.global_stats.matches:,} ({self.global_stats.accuracy:.2f}%)")
        print(f"  Non-Matches: {self.global_stats.non_matches:,}")
        print(f"  Average Similarity: {self.global_stats.avg_similarity:.4f}")
        print(f"  Median Similarity: {self.global_stats.median_similarity:.4f}")
        print(f"  Similarity Range: {self.global_stats.min_similarity:.4f} - {self.global_stats.max_similarity:.4f}")
        print(f"  Std Dev Similarity: {self.global_stats.std_dev_similarity:.4f}")
        print(f"  Unique Cases: {len(self.global_stats.cases_processed)}")
        print(f"  Similarity Basis Distribution:")
        for basis, count in sorted(self.global_stats.similarity_basis_counts.items(), key=lambda x: -x[1]):
            pct = (count / self.global_stats.total_count) * 100
            print(f"    {basis}: {count:,} ({pct:.1f}%)")
        
        print(f"\n📁 STATISTICS BY CATEGORY")
        print("-" * 80)
        for category in sorted(self.category_stats.keys()):
            stats = self.category_stats[category]
            print(f"\n{category}:")
            print(f"  Total: {stats.total_count:,}")
            print(f"  Accuracy: {stats.accuracy:.2f}% ({stats.matches:,} matches, {stats.non_matches:,} non-matches)")
            print(f"  Avg Similarity: {stats.avg_similarity:.4f} (median: {stats.median_similarity:.4f})")
            print(f"  Similarity Range: {stats.min_similarity:.4f} - {stats.max_similarity:.4f}")
            print(f"  Unique Cases: {len(stats.cases_processed)}")
            print(f"  Similarity Basis:")
            for basis, count in sorted(stats.similarity_basis_counts.items(), key=lambda x: -x[1]):
                pct = (count / stats.total_count) * 100
                print(f"    {basis}: {count:,} ({pct:.1f}%)")
        
        print(f"\n🏷️  STATISTICS BY FIELD NAME (Top 20 by count)")
        print("-" * 80)
        sorted_fields = sorted(
            self.field_stats.items(),
            key=lambda x: -x[1].total_count
        )[:20]
        for field_name, stats in sorted_fields:
            print(f"\n{field_name}:")
            print(f"  Total: {stats.total_count:,}")
            print(f"  Accuracy: {stats.accuracy:.2f}% ({stats.matches:,} matches, {stats.non_matches:,} non-matches)")
            print(f"  Avg Similarity: {stats.avg_similarity:.4f} (median: {stats.median_similarity:.4f})")
            print(f"  Similarity Range: {stats.min_similarity:.4f} - {stats.max_similarity:.4f}")
            print(f"  Unique Cases: {len(stats.cases_processed)}")
            if stats.similarity_basis_counts:
                top_basis = max(stats.similarity_basis_counts.items(), key=lambda x: x[1])
                print(f"  Most Common Basis: {top_basis[0]} ({top_basis[1]:,} occurrences)")
        
        print(f"\n📋 STATISTICS BY CATEGORY AND FIELD")
        print("-" * 80)
        for category in sorted(self.category_field_stats.keys()):
            print(f"\n{category}:")
            field_stats = self.category_field_stats[category]
            sorted_cat_fields = sorted(
                field_stats.items(),
                key=lambda x: -x[1].total_count
            )
            for field_name, stats in sorted_cat_fields[:10]:  # Top 10 per category
                print(f"  {field_name}:")
                print(f"    Total: {stats.total_count:,}, Accuracy: {stats.accuracy:.2f}%, "
                      f"Avg Similarity: {stats.avg_similarity:.4f}")
        
        if self.errors:
            print(f"\n⚠️  ERRORS ENCOUNTERED ({len(self.errors)}):")
            for error in self.errors[:10]:  # Show first 10 errors
                print(f"  - {error}")
            if len(self.errors) > 10:
                print(f"  ... and {len(self.errors) - 10} more errors")
        
        print("\n" + "="*80)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze field_comparisons from all master_record.json files"
    )
    parser.add_argument(
        "--base-path",
        type=str,
        default="cp40_processing/output",
        help="Base path to search for master_record.json files (default: cp40_processing/output)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="field_comparison_stats.json",
        help="Output JSON file path (default: field_comparison_stats.json)"
    )
    parser.add_argument(
        "--no-console",
        action="store_true",
        help="Skip console output (only generate JSON file)"
    )
    
    args = parser.parse_args()
    
    analyzer = FieldComparisonAnalyzer(base_path=args.base_path)
    analyzer.analyze()
    
    report = analyzer.generate_report()
    
    # Save JSON report
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n📄 Detailed statistics saved to: {args.output}")
    
    # Print console summary
    if not args.no_console:
        analyzer.print_summary()


if __name__ == "__main__":
    main()

