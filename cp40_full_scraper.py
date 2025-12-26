#!/usr/bin/env python3
"""
Full CP40 Database Scraper
Systematically scrapes the entire CP40 Index Order Search database
and stores results in SQLite with progress tracking and deduplication.

Usage:
    # Start fresh scrape of all years (1349-1596)
    python cp40_full_scraper.py
    
    # Scrape specific year range
    python cp40_full_scraper.py --start-year 1400 --end-year 1410
    
    # Resume interrupted scrape
    python cp40_full_scraper.py --resume
    
    # Show progress/stats only
    python cp40_full_scraper.py --stats
"""

import argparse
import sys
import time
import signal
from datetime import datetime
from typing import Optional

from cp40_surname_scraper_simple import CP40SurnameScraper
from cp40_database import (
    CP40Database, 
    COUNTIES, 
    YEAR_START, 
    YEAR_END,
    SURNAME_PREFIXES
)


class GracefulInterrupt:
    """Handle Ctrl+C gracefully"""
    
    def __init__(self):
        self.interrupted = False
        signal.signal(signal.SIGINT, self._handler)
        signal.signal(signal.SIGTERM, self._handler)
    
    def _handler(self, signum, frame):
        print("\n\n⚠️  Interrupt received. Finishing current job before stopping...")
        self.interrupted = True


class CP40FullScraper:
    """
    Full database scraper that systematically queries year by year
    and stores results in SQLite.
    
    Strategy: Query each year by surname prefix (a*, b*, c*...).
    If a prefix returns 1000 results (server limit), automatically
    split into more granular prefixes (aa*, ab*, ac*...).
    """
    
    # Server returns max 1000 results - if we hit this, need to split further
    MAX_RESULTS_LIMIT = 1000
    
    # Characters for expanding prefixes
    PREFIX_CHARS = "abcdefghijklmnopqrstuvwxyz"
    
    def __init__(
        self, 
        db_path: str = "cp40_records.db",
        delay: float =0.4 ,
        verbose: bool = True
    ):
        """
        Initialize the full scraper
        
        Args:
            db_path: Path to SQLite database
            delay: Delay between requests in seconds
            verbose: Print progress information
        """
        self.db = CP40Database(db_path)
        self.interrupt_handler = GracefulInterrupt()
        # Pass interrupt check to scraper so it can stop during pagination
        # Note: Year extraction is handled by CP40SurnameScraper, which extracts
        # years from the (Term yyyy) pattern in the scraped text (e.g., "Hilary 1400",
        # "Easter 1400", "Trinity 1400", "Michaelmas 1400"), not from CP40/xxxx
        self.scraper = CP40SurnameScraper(
            delay=delay, 
            interrupt_check=lambda: self.interrupt_handler.interrupted
        )
        self.delay = delay
        self.verbose = verbose
    
    def close(self):
        """Clean up resources"""
        self.db.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def log(self, message: str):
        """Print message if verbose mode is on"""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {message}")
    
    def scrape_year(self, year: int, county: Optional[str] = None) -> bool:
        """
        Scrape all entries for a specific year by iterating through
        alphabet prefixes (a*, b*, c*, ...) for granular progress tracking.
        
        Args:
            year: Year to scrape (1349-1596)
            county: Optional county filter (None means all counties)
        
        Returns:
            True if successful, False if failed or interrupted
        """
        county_str = county if county else "All"
        self.log(f"📅 Scraping year {year}, county: {county_str}")
        
        # Check if year already completed
        status = self.db.get_job_status(year, county)
        if status == 'completed':
            self.log(f"  ⏭️  Already completed, skipping")
            return True
        
        # Create year-level job entry
        self.db.create_scrape_job(year, county)
        self.db.start_scrape_job(year, county)
        
        # Initialize prefix jobs for this year
        self.db.initialize_prefix_jobs_for_year(year)
        
        total_results = 0
        total_new = 0
        total_dup = 0
        
        try:
            # Get pending prefixes (allows resuming mid-year)
            pending_prefixes = self.db.get_pending_prefixes_for_year(year)
            
            if not pending_prefixes:
                # All prefixes already done
                progress = self.db.get_year_prefix_progress(year)
                self.log(f"  ⏭️  All prefixes completed ({progress.get('total_new', 0)} entries)")
                self.db.complete_scrape_job(
                    year, county, 
                    progress.get('total_results', 0),
                    progress.get('total_new', 0),
                    0
                )
                return True
            
            completed_count = len(SURNAME_PREFIXES) - len(pending_prefixes)
            if completed_count > 0:
                self.log(f"  🔄 Resuming: {completed_count}/26 prefixes already done")
            
            for prefix in pending_prefixes:
                if self.interrupt_handler.interrupted:
                    self.log(f"  ⚠️  Interrupted at prefix '{prefix}'")
                    self.db.fail_scrape_job(year, county, f"Interrupted at prefix {prefix}")
                    return False
                
                # Scrape this prefix
                new_count, dup_count, result_count = self._scrape_prefix(year, prefix, county)
                
                total_results += result_count
                total_new += new_count
                total_dup += dup_count
                
                # Brief delay between prefix queries
                time.sleep(self.delay * 0.5)
            
            # All prefixes done - mark year as complete
            self.log(f"  ✅ Year {year} complete: {total_results} results, {total_new} new, {total_dup} duplicates")
            self.db.complete_scrape_job(year, county, total_results, total_new, total_dup)
            return True
            
        except Exception as e:
            error_msg = str(e)
            self.log(f"  ❌ Error: {error_msg}")
            self.db.fail_scrape_job(year, county, error_msg)
            return False
    
    def _scrape_prefix(self, year: int, prefix: str, county: Optional[str] = None, depth: int = 0) -> tuple:
        """
        Scrape entries for a specific year and surname prefix.
        If results hit the 1000 limit, recursively split into more granular prefixes.
        
        Args:
            year: Year to scrape
            prefix: Surname prefix (e.g., 'c', 'ca', 'cab')
            county: Optional county filter
            depth: Recursion depth (for logging indentation)
        
        Returns:
            Tuple of (new_count, dup_count, total_results)
        """
        surname_pattern = f"{prefix}*"
        indent = "    " + ("  " * depth)
        
        # Check if already done (only for single-letter prefixes tracked in DB)
        if len(prefix) == 1:
            status = self.db.get_prefix_job_status(year, prefix)
            if status == 'completed':
                return 0, 0, 0
            self.db.create_prefix_job(year, prefix)
            self.db.start_prefix_job(year, prefix)
        
        try:
            results = self.scraper.search_surname(
                surname=surname_pattern,
                year_from=str(year),
                year_to=str(year),
                county=county if county else "All"
            )
            
            # Check for interrupt during pagination
            if self.interrupt_handler.interrupted:
                # Save what we got before stopping
                if results:
                    new_count, dup_count = self.db.insert_entries(results)
                    self.log(f"{indent}[{prefix}*] Interrupted, saved {new_count} entries")
                if len(prefix) == 1:
                    self.db.fail_prefix_job(year, prefix, "Interrupted")
                return 0, 0, 0
            
            if not results:
                if len(prefix) == 1:
                    self.db.complete_prefix_job(year, prefix, 0, 0, 0)
                return 0, 0, 0
            
            # Check if we hit the 1000 result limit - need to split further
            if len(results) >= self.MAX_RESULTS_LIMIT:
                self.log(f"{indent}[{prefix}*] Hit {len(results)} limit! Splitting into {prefix}a*, {prefix}b*...")
                
                # Save what we have first (there may be some unique results)
                new_count, dup_count = self.db.insert_entries(results)
                total_new = new_count
                total_dup = dup_count
                total_results = len(results)
                
                # Now query more specific prefixes to get records we missed
                for char in self.PREFIX_CHARS:
                    if self.interrupt_handler.interrupted:
                        break
                    
                    sub_prefix = prefix + char
                    sub_new, sub_dup, sub_total = self._scrape_prefix(
                        year, sub_prefix, county, depth + 1
                    )
                    total_new += sub_new
                    total_dup += sub_dup
                    total_results += sub_total
                    
                    time.sleep(self.delay * 0.3)  # Brief delay between sub-queries
                
                if len(prefix) == 1:
                    self.db.complete_prefix_job(year, prefix, total_results, total_new, total_dup)
                
                return total_new, total_dup, total_results
            
            # Normal case: under limit, save results
            new_count, dup_count = self.db.insert_entries(results)
            
            if new_count > 0:
                self.log(f"{indent}[{prefix}*] {len(results)} results: {new_count} new, {dup_count} dup")
            
            if len(prefix) == 1:
                self.db.complete_prefix_job(year, prefix, len(results), new_count, dup_count)
            
            return new_count, dup_count, len(results)
            
        except Exception as e:
            if len(prefix) == 1:
                self.db.fail_prefix_job(year, prefix, str(e))
            raise
    
    def _scrape_year_by_county(self, year: int) -> bool:
        """
        Scrape a year by querying each county separately
        (used when a single year has too many results)
        """
        self.log(f"  Splitting year {year} into {len(COUNTIES) - 1} county queries")
        
        success = True
        for county in COUNTIES:
            if county == "All":
                continue
            
            if self.interrupt_handler.interrupted:
                self.log("  Interrupted!")
                return False
            
            # Check if this county was already done
            status = self.db.get_job_status(year, county)
            if status == 'completed':
                continue
            
            if not self.scrape_year(year, county):
                success = False
            
            time.sleep(self.delay)
        
        # Mark the "All" job as completed if all counties succeeded
        if success:
            self.db.complete_scrape_job(year, None, 0, 0, 0)
        
        return success
    
    def scrape_range(
        self, 
        start_year: int = YEAR_START, 
        end_year: int = YEAR_END,
        resume: bool = False
    ) -> dict:
        """
        Scrape a range of years
        
        Args:
            start_year: First year to scrape
            end_year: Last year to scrape (inclusive)
            resume: If True, skip already completed years
        
        Returns:
            Dict with statistics about the scrape
        """
        stats = {
            'years_attempted': 0,
            'years_completed': 0,
            'years_failed': 0,
            'years_skipped': 0,
            'interrupted': False
        }
        
        # Initialize jobs for the range
        self.log(f"🚀 Starting scrape for years {start_year}-{end_year}")
        self.db.initialize_year_jobs(start_year, end_year)
        
        for year in range(start_year, end_year + 1):
            if self.interrupt_handler.interrupted:
                self.log("🛑 Interrupted by user")
                stats['interrupted'] = True
                break
            
            # Check if should skip
            if resume:
                status = self.db.get_job_status(year, None)
                if status == 'completed':
                    stats['years_skipped'] += 1
                    continue
            
            stats['years_attempted'] += 1
            
            if self.scrape_year(year):
                stats['years_completed'] += 1
            else:
                stats['years_failed'] += 1
            
            # Be polite to the server
            time.sleep(self.delay)
        
        return stats
    
    def resume_scrape(self) -> dict:
        """Resume scraping from where we left off"""
        self.log("🔄 Resuming previous scrape...")
        
        # Get the year range from existing jobs
        incomplete = self.db.get_incomplete_jobs()
        if not incomplete:
            self.log("✅ No incomplete jobs found!")
            return {'years_attempted': 0}
        
        years = sorted(set(job['year'] for job in incomplete))
        self.log(f"Found {len(years)} years with incomplete jobs")
        
        return self.scrape_range(
            start_year=min(years),
            end_year=max(years),
            resume=True
        )
    
    def show_stats(self):
        """Display database statistics"""
        stats = self.db.get_stats()
        
        print("\n" + "═" * 60)
        print("📊 CP40 DATABASE STATISTICS")
        print("═" * 60)
        
        print(f"\n📝 ENTRIES")
        print(f"   Total entries: {stats['total_entries']:,}")
        if stats['min_year'] and stats['max_year']:
            print(f"   Year range: {stats['min_year']} - {stats['max_year']}")
        
        print(f"\n👤 PERSONS: {stats['total_persons']:,}")
        print(f"📍 PLACES: {stats['total_places']:,}")
        
        print(f"\n🔄 YEAR-LEVEL JOBS")
        jobs = stats.get('jobs_by_status', {})
        total_jobs = sum(jobs.values())
        print(f"   Total years: {total_jobs}")
        for status, count in sorted(jobs.items()):
            pct = (count / total_jobs * 100) if total_jobs > 0 else 0
            emoji = {'completed': '✅', 'pending': '⏳', 'failed': '❌', 'in_progress': '🔄'}.get(status, '❓')
            print(f"   {emoji} {status}: {count} ({pct:.1f}%)")
        
        # Show prefix job stats
        prefix_stats = stats.get('prefix_jobs_by_status', {})
        if prefix_stats:
            total_prefix = sum(prefix_stats.values())
            print(f"\n🔤 PREFIX-LEVEL JOBS")
            print(f"   Total prefix jobs: {total_prefix}")
            for status, count in sorted(prefix_stats.items()):
                pct = (count / total_prefix * 100) if total_prefix > 0 else 0
                emoji = {'completed': '✅', 'pending': '⏳', 'failed': '❌', 'in_progress': '🔄'}.get(status, '❓')
                print(f"   {emoji} {status}: {count} ({pct:.1f}%)")
        
        if stats.get('top_counties'):
            print(f"\n🏛️ TOP COUNTIES")
            for county, count in stats['top_counties'][:5]:
                county_name = county if county else "(unknown)"
                print(f"   {county_name}: {count:,}")
        
        print("\n" + "═" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Scrape the entire CP40 database into SQLite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start fresh scrape of all years
  python cp40_full_scraper.py
  
  # Scrape specific year range
  python cp40_full_scraper.py --start-year 1400 --end-year 1410
  
  # Resume interrupted scrape  
  python cp40_full_scraper.py --resume
  
  # Just show statistics
  python cp40_full_scraper.py --stats

Notes:
  - The scraper uses a wildcard (*) surname search to get all entries for each year
  - Progress is saved to SQLite, so you can safely interrupt and resume
  - Default delay is 1 second between requests (be polite to the server!)
  - Estimated time for full scrape: several hours to days depending on data volume
        """
    )
    
    parser.add_argument(
        '--db',
        default='cp40_records.db',
        help='SQLite database path (default: cp40_records.db)'
    )
    parser.add_argument(
        '--start-year',
        type=int,
        default=YEAR_START,
        help=f'Starting year (default: {YEAR_START})'
    )
    parser.add_argument(
        '--end-year',
        type=int,
        default=YEAR_END,
        help=f'Ending year (default: {YEAR_END})'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=1.0,
        help='Delay between requests in seconds (default: 1.0)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from where we left off (skip completed years)'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show database statistics and exit'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )
    
    args = parser.parse_args()
    
    # Validate year range
    if args.start_year < YEAR_START or args.end_year > YEAR_END:
        print(f"Error: Year range must be between {YEAR_START} and {YEAR_END}")
        return 1
    
    if args.start_year > args.end_year:
        print("Error: start-year must be <= end-year")
        return 1
    
    try:
        with CP40FullScraper(
            db_path=args.db,
            delay=args.delay,
            verbose=not args.quiet
        ) as scraper:
            
            if args.stats:
                scraper.show_stats()
                return 0
            
            if args.resume:
                stats = scraper.resume_scrape()
            else:
                stats = scraper.scrape_range(
                    start_year=args.start_year,
                    end_year=args.end_year,
                    resume=False
                )
            
            # Show final statistics
            print("\n" + "═" * 60)
            print("📈 SCRAPE SUMMARY")
            print("═" * 60)
            print(f"   Years attempted: {stats['years_attempted']}")
            print(f"   Years completed: {stats['years_completed']}")
            print(f"   Years failed: {stats['years_failed']}")
            print(f"   Years skipped: {stats['years_skipped']}")
            if stats.get('interrupted'):
                print("   ⚠️  Scrape was interrupted (use --resume to continue)")
            
            scraper.show_stats()
            
            return 0 if not stats.get('interrupted') else 130
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

