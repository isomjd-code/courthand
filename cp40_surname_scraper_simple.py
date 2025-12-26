#!/usr/bin/env python3
"""
Simple scraper for Medieval Genealogy CP40 Index Order Search
Searches for surnames and scrapes results from:
https://www.medievalgenealogy.org.uk/aalt/cp40_search.php
"""

import requests
from bs4 import BeautifulSoup
import time
import json
import re
from typing import List, Dict, Optional
from urllib.parse import urljoin
import argparse


class CP40SurnameScraper:
    """Scraper for CP40 surname search results"""
    
    BASE_URL = "https://www.medievalgenealogy.org.uk/aalt/cp40_search.php"
    
    def __init__(self, delay: float = 1.0, interrupt_check: callable = None):
        """
        Initialize the scraper
        
        Args:
            delay: Delay between requests in seconds (to be polite to the server)
            interrupt_check: Optional callable that returns True if scraping should stop
        """
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.delay = delay
        self.interrupt_check = interrupt_check
    
    def search_surname(
        self,
        surname: str,
        forename: str = "",
        place: str = "",
        year_from: str = "",
        year_to: str = "",
        county: str = "All",
        soundex_surname: bool = False
    ) -> List[Dict]:
        """
        Search for a surname and return all results
        
        Args:
            surname: Surname to search for (supports wildcards: ? for single char, * for multiple)
            forename: Optional forename filter
            place: Optional place filter
            year_from: Starting year
            year_to: Ending year
            county: County filter (default: "All")
            soundex_surname: Use Soundex for surname matching
        
        Returns:
            List of dictionaries containing parsed result data
        """
        all_results = []
        next_url = None
        page_num = 1
        
        print(f"Searching for surname: {surname}")
        
        while True:
            # Check for interrupt request
            if self.interrupt_check and self.interrupt_check():
                print("  Scraping interrupted by user")
                break
            
            print(f"  Fetching page {page_num}...")
            
            if page_num == 1:
                # First page: submit the search form
                results, next_url = self._submit_search(
                    surname=surname,
                    forename=forename,
                    place=place,
                    year_from=year_from,
                    year_to=year_to,
                    county=county,
                    soundex_surname=soundex_surname
                )
            else:
                # Subsequent pages: follow the next link
                if not next_url:
                    break
                results, next_url = self._fetch_results_page(next_url)
            
            if not results:
                if page_num == 1:
                    print("  No results found")
                break
            
            all_results.extend(results)
            print(f"  Found {len(results)} results on page {page_num} (total: {len(all_results)})")
            
            if not next_url:
                break
            
            page_num += 1
            time.sleep(self.delay)
        
        print(f"\nTotal results found: {len(all_results)}")
        return all_results
    
    def _submit_search(
        self,
        surname: str,
        forename: str,
        place: str,
        year_from: str,
        year_to: str,
        county: str,
        soundex_surname: bool
    ) -> tuple[List[Dict], Optional[str]]:
        """Submit the search form and return results"""
        
        # Prepare form parameters - the form uses GET method!
        # Field names from the actual form:
        # - surname (lowercase)
        # - forename (lowercase)
        # - place (lowercase)
        # - after (for year from)
        # - before (for year to)
        # - county (lowercase)
        # - s=1 (hidden field, required!)
        
        params = {
            's': '1',  # Required hidden field
            'surname': surname,
            'forename': forename,
            'place': place,
        }
        
        if year_from:
            params['after'] = year_from  # Note: 'after' not 'Year'
        if year_to:
            params['before'] = year_to  # Note: 'before' not 'YearTo'
        if county and county != 'All':
            params['county'] = county
        
        if soundex_surname:
            params['soundexsurname'] = '1'  # Note: value is '1' not 'on'
        
        # Submit GET request (not POST!)
        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"  Error submitting search: {e}")
            return [], None
        
        # Parse results
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Check for error message
        if soup.find(string=re.compile(r'Please enter some search terms', re.I)):
            return [], None
        
        # Parse results and find next page link
        results = self._parse_results(soup)
        next_url = self._find_next_link(soup)
        
        return results, next_url
    
    def _fetch_results_page(self, url: str) -> tuple[List[Dict], Optional[str]]:
        """Fetch a results page from a URL"""
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"  Error fetching page: {e}")
            return [], None
        
        soup = BeautifulSoup(response.content, 'html.parser')
        results = self._parse_results(soup)
        next_url = self._find_next_link(soup)
        
        return results, next_url
    
    def _parse_results(self, soup: BeautifulSoup) -> List[Dict]:
        """Parse the HTML to extract result entries"""
        results = []
        
        # Results are typically in an ordered list <ol>
        result_list = soup.find('ol')
        if result_list:
            list_items = result_list.find_all('li', recursive=False)
            
            for li in list_items:
                entry = self._parse_list_item(li)
                if entry:
                    results.append(entry)
        else:
            # Try alternative structures (table, divs, etc.)
            # Check for any structured data that might be results
            result_divs = soup.find_all(['div', 'tr', 'p'], 
                                      class_=re.compile(r'result|entry|item', re.I))
            
            if not result_divs:
                # Look for any repeated structure that might be results
                # This is a fallback - may need adjustment based on actual HTML
                pass
        
        return results
    
    def _parse_list_item(self, li) -> Optional[Dict]:
        """Parse a single <li> entry from the results"""
        entry = {
            'raw_text': '',
            'roll_reference': '',
            'links': [],
            'county': None,
            'persons': [],
            'places': [],
            'year': None
        }
        
        # Get full text
        full_text = li.get_text(' ', strip=True)
        entry['raw_text'] = full_text
        
        if not full_text:
            return None
        
        # Extract links
        for link in li.find_all('a', href=True):
            href = link.get('href')
            link_text = link.get_text(strip=True)
            
            if href:
                # Make URL absolute if relative
                if not href.startswith('http'):
                    href = urljoin(self.BASE_URL, href)
                
                entry['links'].append({
                    'url': href,
                    'text': link_text
                })
        
        # Extract roll reference (usually before first bracket)
        ref_match = re.match(r'([^\[]+?)\s*\[', full_text)
        if ref_match:
            entry['roll_reference'] = ref_match.group(1).strip()
        else:
            # Fallback: first part of text
            entry['roll_reference'] = full_text.split('[')[0].strip() if '[' in full_text else full_text[:100]
        
        # Extract County
        county_match = re.search(r'County:\s*([^;]+?)(?:;|$)', full_text)
        if county_match:
            entry['county'] = county_match.group(1).strip()
        
        # Extract Persons (capture everything until "Places:" or end)
        persons_match = re.search(r'Persons:\s*(.+?)(?:\s*;\s*Places:|$)', full_text)
        if persons_match:
            persons_str = persons_match.group(1).strip()
            # Split by semicolon - persons are separated by semicolons
            entry['persons'] = [p.strip() for p in persons_str.split(';') if p.strip()]
        
        # Extract Places (capture everything until end or next semicolon-separated field)
        places_match = re.search(r'Places:\s*(.+?)$', full_text)
        if places_match:
            places_str = places_match.group(1).strip()
            # Split by semicolon - places are separated by semicolons
            entry['places'] = [p.strip() for p in places_str.split(';') if p.strip()]
        
        # Extract year from (Term yyyy) pattern
        # Legal terms: Hilary, Easter, Trinity, Michaelmas
        # Look for pattern like "(Hilary 1400)" or "(Michaelmas 1450)" or "(Easter: 1400)"
        # Pattern allows for optional colon and whitespace variations
        year_match = re.search(
            r'\((?:Hilary|Easter|Trinity|Michaelmas)\s*:?\s*(\d{4})\)', 
            full_text, 
            re.IGNORECASE
        )
        if year_match:
            entry['year'] = year_match.group(1)
        else:
            # Fallback: try to find any 4-digit year (1300-1600) if (Term yyyy) not found
            year_match = re.search(r'\b(1[3-5]\d{2})\b', full_text)
            if year_match:
                entry['year'] = year_match.group(1)
        
        return entry
    
    def _find_next_link(self, soup: BeautifulSoup) -> Optional[str]:
        """Find the URL for the next page of results"""
        # Look for "Next" link
        next_link = soup.find('a', string=re.compile(r'^Next\s*$', re.I))
        
        if next_link and next_link.get('href'):
            href = next_link.get('href')
            if not href.startswith('http'):
                href = urljoin(self.BASE_URL, href)
            return href
        
        # Alternative: look for next in pagination
        pagination = soup.find(class_=re.compile(r'pagination|nav', re.I))
        if pagination:
            next_links = pagination.find_all('a', string=re.compile(r'Next|next|>', re.I))
            if next_links:
                href = next_links[0].get('href')
                if href:
                    if not href.startswith('http'):
                        href = urljoin(self.BASE_URL, href)
                    return href
        
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Scrape surname search results from CP40 Index Order Search',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple surname search
  python cp40_surname_scraper_simple.py Smith
  
  # Search with year range
  python cp40_surname_scraper_simple.py Smith --year-from 1400 --year-to 1450
  
  # Search with wildcards
  python cp40_surname_scraper_simple.py "Sm*" --county Yorkshire
  
  # Use Soundex matching
  python cp40_surname_scraper_simple.py Smyth --soundex
        """
    )
    
    parser.add_argument(
        'surname',
        help='Surname to search for (supports wildcards: ? for single char, * for multiple)'
    )
    parser.add_argument(
        '--forename',
        default='',
        help='Optional forename filter'
    )
    parser.add_argument(
        '--place',
        default='',
        help='Optional place filter'
    )
    parser.add_argument(
        '--year-from',
        default='',
        help='Starting year (e.g., 1349)'
    )
    parser.add_argument(
        '--year-to',
        default='',
        help='Ending year (e.g., 1596)'
    )
    parser.add_argument(
        '--county',
        default='All',
        help='County filter (default: All)'
    )
    parser.add_argument(
        '--soundex',
        action='store_true',
        help='Use Soundex matching for surname'
    )
    parser.add_argument(
        '--output',
        default='cp40_results.json',
        help='Output JSON file (default: cp40_results.json)'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=1.0,
        help='Delay between requests in seconds (default: 1.0)'
    )
    
    args = parser.parse_args()
    
    scraper = CP40SurnameScraper(delay=args.delay)
    
    try:
        results = scraper.search_surname(
            surname=args.surname,
            forename=args.forename,
            place=args.place,
            year_from=args.year_from,
            year_to=args.year_to,
            county=args.county,
            soundex_surname=args.soundex
        )
        
        # Save results to JSON
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to {args.output}")
        
        # Print summary
        if results:
            print(f"\nSample result:")
            print(json.dumps(results[0], indent=2, ensure_ascii=False)[:500] + "...")
        
    except KeyboardInterrupt:
        print("\n\nSearch interrupted by user")
        return 1
    except requests.RequestException as e:
        print(f"\nError making request: {e}")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

