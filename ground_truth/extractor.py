"""Database extraction logic for ground truth records."""

from __future__ import annotations

import json
import os
import sqlite3
from typing import List, Optional

from .config import DB_PATH, OUTPUT_DIR, TARGET_ROLL, TARGET_ROTULUS
from .query import JSON_QUERY


def _ensure_output_dir(path: str) -> None:
    """
    Ensure that the output directory exists, creating it if necessary.

    Args:
        path: Directory path to check and create if missing.
    """
    os.makedirs(path, exist_ok=True)


def extract_case_data(
    roll: str = TARGET_ROLL,
    rotulus: str = TARGET_ROTULUS,
    *,
    db_path: str = DB_PATH,
    output_dir: str = OUTPUT_DIR,
) -> List[dict]:
    """
    Extract case data from the CP40 database for specified roll and rotulus numbers.

    Queries the database for cases matching the roll and rotulus criteria, then
    extracts comprehensive case information including:
    - Case details (county, damages, writ type, term)
    - Parties (names, roles, occupations, status, locations)
    - Events (types, values, locations, dates)
    - Pleadings and postea outcomes

    Each matching case is saved as a JSON file in the output directory with
    a filename based on the case ID and reference.

    Args:
        roll: Roll number to search for (e.g., "562"). Defaults to TARGET_ROLL.
        rotulus: Rotulus number to search for (e.g., "340"). Defaults to TARGET_ROTULUS.
        db_path: Path to the SQLite database file. Defaults to DB_PATH.
        output_dir: Directory where JSON files will be saved. Defaults to OUTPUT_DIR.

    Returns:
        A list of JSON data dictionaries for all matching cases, or an empty list if:
        - The database file does not exist
        - No matching cases are found
        - A database error occurs
        - All cases fail to generate valid JSON

    Note:
        The function uses exact matching for roll and rotulus. Multiple cases may be
        found if there are multiple distinct cases with the same roll/rotulus combination.
        Duplicate cases from multiple references are filtered out using DISTINCT.
        
        Fallback behavior: If the rotulus ends with a letter (e.g., "277d" for dorse)
        and no exact match is found, the function will automatically try the recto version
        (e.g., "277") as a fallback. This allows matching dorse-side cases to their
        corresponding recto-side ground truth when the dorse side hasn't been entered yet.
    """
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        return []

    _ensure_output_dir(output_dir)
    conn = None
    all_cases: List[dict] = []
    seen_case_ids = set()  # Track processed CaseIDs to avoid duplicates

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        print(f"Searching for Roll: {roll}, Rotulus: {rotulus}...")

        # Use exact matching for both roll and rotulus
        # r.reference is just the roll number (e.g., "565" or 565)
        # c.CaseRot is the rotulus number (e.g., "123" or 123)
        # Use DISTINCT to avoid duplicate cases from multiple references
        # Handle leading zeros: database may store "086" when searching for "86", or "004" when searching for "4"
        # Generate possible formats for rotulus: original and padded versions
        rotulus_formats_set = {rotulus}
        # Try padding to 3 digits (e.g., "86" -> "086", "4" -> "004")
        if rotulus and rotulus.isdigit():
            rotulus_formats_set.add(rotulus.zfill(3))
            # Also try 2 digits for 1-digit numbers (e.g., "4" -> "04")
            if len(rotulus) == 1:
                rotulus_formats_set.add(rotulus.zfill(2))
        rotulus_formats = list(rotulus_formats_set)
        
        # Build query with OR conditions for different rotulus formats
        placeholders = ",".join(["?" for _ in rotulus_formats])
        lookup_query = f"""
            SELECT DISTINCT c.CaseID, c.CaseRot, r.reference
            FROM TblCase c
            JOIN TblReference r ON c.DocID = r.docid
            WHERE CAST(r.reference AS TEXT) = ? 
            AND CAST(c.CaseRot AS TEXT) IN ({placeholders})
        """
        
        cursor.execute(lookup_query, (roll, *rotulus_formats))
        matches = cursor.fetchall()

        # Fallback: if no matches found and rotulus ends with a letter (e.g., "d" for dorse),
        # try querying without the suffix (e.g., "277d" -> "277" for recto)
        if not matches and rotulus and rotulus[-1].isalpha():
            rotulus_fallback = rotulus[:-1]  # Strip trailing letter
            print(f"No matches found for {rotulus}, trying fallback to recto version: {rotulus_fallback}...")
            # Generate possible formats for fallback rotulus (with leading zero handling)
            fallback_formats_set = {rotulus_fallback}
            if rotulus_fallback and rotulus_fallback.isdigit():
                fallback_formats_set.add(rotulus_fallback.zfill(3))
                if len(rotulus_fallback) == 1:
                    fallback_formats_set.add(rotulus_fallback.zfill(2))
            fallback_formats = list(fallback_formats_set)
            
            fallback_placeholders = ",".join(["?" for _ in fallback_formats])
            fallback_query = f"""
                SELECT DISTINCT c.CaseID, c.CaseRot, r.reference
                FROM TblCase c
                JOIN TblReference r ON c.DocID = r.docid
                WHERE CAST(r.reference AS TEXT) = ? 
                AND CAST(c.CaseRot AS TEXT) IN ({fallback_placeholders})
            """
            cursor.execute(fallback_query, (roll, *fallback_formats))
            matches = cursor.fetchall()
            if matches:
                print(f"Found {len(matches)} case(s) using fallback rotulus {rotulus_fallback}")

        if not matches:
            print("No cases found matching those criteria.")
            return []

        print(f"Found {len(matches)} unique case(s). Generating JSON...")
        for case_id, case_rot, ref_text in matches:
            # Skip if we've already processed this CaseID
            if case_id in seen_case_ids:
                continue
            seen_case_ids.add(case_id)
            
            cursor.execute(JSON_QUERY, (case_id,))
            row = cursor.fetchone()

            if not row or not row[0]:
                print(f"Failed to generate JSON for CaseID {case_id}")
                continue

            try:
                json_data = json.loads(row[0])
            except json.JSONDecodeError:
                print(f"Error: SQL returned malformed JSON for CaseID {case_id}")
                continue

            safe_ref = ref_text.replace(" ", "_").replace("/", "-")
            filename = f"GroundTruth_CaseID{case_id}_{safe_ref}.json"
            filepath = os.path.join(output_dir, filename)

            with open(filepath, "w", encoding="utf-8") as handle:
                json.dump(json_data, handle, indent=2, ensure_ascii=False)

            print(f"Saved: {filepath}")
            all_cases.append(json_data)

        return all_cases
    except sqlite3.Error as error:
        print(f"SQLite Error: {error}")
        return []
    finally:
        if conn:
            conn.close()
            print("Done.")


def main() -> None:
    """
    CLI entry point for extracting ground truth data.

    Extracts case data using the default roll and rotulus numbers specified
    in the configuration. This is the main entry point when running the
    ground_truth module as a script.

    The default values are:
    - Roll: From TARGET_ROLL config (default: "562")
    - Rotulus: From TARGET_ROTULUS config (default: "340")
    - Database: From DB_PATH config
    - Output: From OUTPUT_DIR config (default: "ground_truth_output")
    """
    extract_case_data()

