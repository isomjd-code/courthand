import time
import json
import string
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

# Configuration
BASE_URL = "http://www.medievalgenealogy.org.uk/cp40_search_alpha.php"
OUTPUT_FILE = "plea_rolls_data.json"

# specific logic to parse the text content of a result entry
def parse_entry_html(li_element):
    """
    Parses a single <li> element from the search results.
    """
    entry_data = {
        "raw_text": "",
        "roll_reference": "",
        "links": [],
        "county": None,
        "persons": [],
        "places": []
    }

    # Use BeautifulSoup to parse the inner HTML of the Selenium element
    html = li_element.get_attribute('innerHTML')
    soup = BeautifulSoup(html, 'html.parser')
    
    # Get raw text for cleanup
    full_text = soup.get_text(" ", strip=True)
    entry_data["raw_text"] = full_text

    # 1. Extract Links (AALT Index and Image)
    links = []
    for a in soup.find_all('a', href=True):
        href = a['href']
        # Ensure absolute URLs
        if not href.startswith('http'):
            # The site uses relative links, we need to construct the full path
            # Based on the provided HTML, links look like 'cp40_link.php?...'
            href = "http://www.medievalgenealogy.org.uk/" + href.lstrip('/')
        links.append(href)
    entry_data["links"] = links

    # 2. Extract Roll Reference (Text before the first <small> or <br>)
    # The structure is usually: Text Node (Ref) <small>Links</small> <br> <small>Details</small>
    # We can split the full text by "Index at AALT" or just take the text up to the bracket
    match_ref = re.match(r"(.*?)\s*\[", full_text)
    if match_ref:
        entry_data["roll_reference"] = match_ref.group(1).strip()
    else:
        # Fallback if structure varies
        entry_data["roll_reference"] = full_text.split('[')[0].strip()

    # 3. Extract Structured Data (County, Persons, Places)
    # The details are typically in the text after the links.
    # We look for specific keywords: "County:", "Persons:", "Places:"
    
    details_text = full_text
    
    # Extract County
    county_match = re.search(r"County:\s*(.*?)(?:;|$)", details_text)
    if county_match:
        entry_data["county"] = county_match.group(1).strip()

    # Extract Persons
    # Persons usually end at a semicolon (before Places) or end of string
    persons_match = re.search(r"Persons:\s*(.*?)(?:;\s*Places:|;|$)", details_text)
    if persons_match:
        p_str = persons_match.group(1).strip()
        # Split by semicolon if multiple persons are listed with delimiters, 
        # though the example shows them separated by semicolons in the bold text or just spaces/commas.
        # Looking at example: "William Aas; Nicholas Clerk..."
        entry_data["persons"] = [p.strip() for p in p_str.split(';')]

    # Extract Places
    places_match = re.search(r"Places:\s*(.*?)(?:;|$)", details_text)
    if places_match:
        pl_str = places_match.group(1).strip()
        entry_data["places"] = [p.strip() for p in pl_str.split(';')]

    return entry_data

def main():
    # Setup Headless Chrome
    chrome_options = Options()
    chrome_options.add_argument("--headless") 
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    all_results = []

    # Generate search patterns: aa*, ab*, ... az*
    # You can expand this loop to 'ba*', 'bb*' etc., if needed.
    # Currently set to 'aa*' through 'az*'
    search_patterns = [f"a{char}*" for char in string.ascii_lowercase]

    try:
        for surname_pattern in search_patterns:
            print(f"Searching for surname: {surname_pattern}...")
            
            # Go to search page
            driver.get(BASE_URL)
            
            try:
                # Wait for form to load
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.NAME, "surname"))
                )

                # Fill form
                surname_input = driver.find_element(By.NAME, "surname")
                surname_input.clear()
                surname_input.send_keys(surname_pattern)

                # It is good practice to uncheck Soundex if using wildcards, 
                # though the site says it ignores it automatically.
                # soundex_cb = driver.find_element(By.NAME, "soundexsurname")
                # if soundex_cb.is_selected():
                #     soundex_cb.click()

                # Submit form
                submit_btn = driver.find_element(By.CSS_SELECTOR, "input[value='Search']")
                submit_btn.click()

                # --- Pagination Loop ---
                while True:
                    # Wait for results to load
                    try:
                        # Check if results exist (look for the <ol> tag)
                        WebDriverWait(driver, 5).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, "ol"))
                        )
                    except:
                        # If timeout, likely 0 results found for this pattern
                        print(f"  No results found for {surname_pattern}")
                        break

                    # Find all list items in the ordered list
                    result_items = driver.find_elements(By.CSS_SELECTOR, "ol > li")
                    
                    print(f"  Scraping {len(result_items)} items on current page...")

                    for item in result_items:
                        try:
                            data = parse_entry_html(item)
                            data['search_pattern'] = surname_pattern
                            all_results.append(data)
                        except Exception as e:
                            print(f"  Error parsing item: {e}")

                    # Check for "Next" button
                    try:
                        # The HTML shows the next link has text "Next"
                        next_link = driver.find_element(By.LINK_TEXT, "Next")
                        next_url = next_link.get_attribute("href")
                        
                        # Navigate to next page
                        driver.get(next_url)
                        time.sleep(1) # Be polite to the server
                    except:
                        # No "Next" link found, break pagination loop
                        break
                
            except Exception as e:
                print(f"Error processing pattern {surname_pattern}: {e}")
                continue
            
            # Be polite between search queries
            time.sleep(2)

    finally:
        driver.quit()

    # Save to JSON
    print(f"Scraping complete. Found {len(all_results)} total entries.")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    print(f"Data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()