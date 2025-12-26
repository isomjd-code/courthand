import sqlite3
import json
import os
import argparse
from typing import List, Dict, Any

class SchemaGenerator:
    def __init__(self, db_path):
        self.db_path = db_path
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database not found at: {db_path}")
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()

    def get_enum_values(self, table: str, column: str, limit: int = 100) -> List[str]:
        """
        Fetches distinct values from a column to build an ENUM list.
        It prioritizes frequency (most used values first).
        """
        try:
            # Check if table exists
            self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
            if not self.cursor.fetchone():
                return []

            # Specific handling for quoted table names
            safe_table = f'"{table}"'
            safe_col = f'"{column}"'
            
            query = f"""
                SELECT {safe_col}, COUNT(*) as freq 
                FROM {safe_table} 
                WHERE {safe_col} IS NOT NULL AND {safe_col} != ''
                GROUP BY {safe_col} 
                ORDER BY freq DESC 
                LIMIT ?
            """
            self.cursor.execute(query, (limit,))
            return [row[0] for row in self.cursor.fetchall()]
        except sqlite3.Error as e:
            print(f"Warning: Could not query {table}.{column}: {e}")
            return []

    def generate(self):
        print(f"Analyzing database: {self.db_path}...")

        # --- 1. EXTRACT DATA FOR ENUMS ---
        
        # Writ Types
        # Prioritize LookUp, fallback to transaction table
        writ_types = self.get_enum_values("LookUpWritType(sub-type)", "Writ Type (sub-type)")
        if not writ_types:
            writ_types = self.get_enum_values("TblWritType(sub-type)", "Writ type (sub-type)")

        # Roles
        roles = self.get_enum_values("LookUpIndividualRole", "role")
        if not roles:
            roles = self.get_enum_values("TblAgentRole", "role")

        # Statuses / Social Markers
        statuses = self.get_enum_values("LookUpStatus", "status")
        if not statuses:
            statuses = self.get_enum_values("TblAgentStatus", "AgentStatus")

        # Occupations
        occupations = self.get_enum_values("TblOccupation", "AgentOccupation")
        
        # Counties
        counties = self.get_enum_values("LookUpCounty", "county")
        
        # Currencies (Likely free text, but let's see common formats)
        # We don't make this an enum, but we can sample it for description
        damages_sample = self.get_enum_values("TblCase", "DamClaimed", limit=5)

        # Terms
        terms = ["Michaelmas", "Hilary", "Easter", "Trinity"] # Standard, but could query TblReference

        # --- 2. BUILD THE SCHEMA OBJECT ---
        
        schema = {
            "type": "OBJECT",
            "properties": {
                "cases": {
                    "type": "ARRAY",
                    "items": {"$ref": "#/$defs/caseRecord"}
                }
            },
            "required": ["cases"],
            "$defs": {
                # --- DEFINITIONS ---
                
                "currency": {
                    "type": "OBJECT",
                    "description": f"Standard format example found in DB: {damages_sample}",
                    "properties": {
                        "original_string": {"type": "STRING"},
                        "pence_value": {"type": "NUMBER", "description": "Calculated value in pence"},
                        "shillings_approx": {"type": "NUMBER"}
                    },
                    "required": ["original_string"]
                },

                "agent": {
                    "type": "OBJECT",
                    "properties": {
                        "gender": {"type": "STRING", "enum": ["m", "f", "unknown"]},
                        "life_status": {"type": "STRING", "enum": ["living", "deceased"]},
                        "name": {
                            "type": "OBJECT",
                            "properties": {
                                "christian_name": {"type": "STRING"},
                                "surname": {"type": "STRING"},
                                "suffix": {"type": "STRING"}
                            }
                        },
                        "institution": {
                            "type": "OBJECT",
                            "properties": {
                                "name": {"type": "STRING"},
                                "type": {"type": "STRING"}
                            }
                        },
                        "occupations": {
                            "type": "ARRAY",
                            "items": {
                                "type": "STRING",
                                "enum": occupations, 
                                "description": "Select from known database occupations if possible"
                            }
                        },
                        "status_markers": {
                            "type": "ARRAY",
                            "items": {
                                "type": "STRING",
                                "enum": statuses,
                                "description": "Social status markers"
                            }
                        },
                        "roles": {
                            "type": "ARRAY",
                            "items": {
                                "type": "STRING",
                                "enum": roles,
                                "description": "Legal role in the case"
                            }
                        },
                        "location": {
                            "type": "OBJECT",
                            "properties": {
                                "country": {"type": "STRING"},
                                "county": {"type": "STRING", "enum": counties},
                                "parish": {"type": "STRING"},
                                "ward": {"type": "STRING"}
                            }
                        }
                    },
                    "required": ["name", "roles"]
                },

                "caseRecord": {
                    "type": "OBJECT",
                    "properties": {
                        "reference": {
                            "type": "OBJECT",
                            "properties": {
                                "roll_number": {"type": "INTEGER"},
                                "term": {"type": "STRING", "enum": terms},
                                "year_calendar": {"type": "INTEGER"},
                                "rotulet": {"type": "STRING"},
                                "record_series": {"type": "STRING", "const": "CP 40"}
                            },
                            "required": ["roll_number", "year_calendar"]
                        },
                        "case_data": {
                            "type": "OBJECT",
                            "properties": {
                                "county_marginal": {"type": "STRING", "enum": counties},
                                "writ_type": {
                                    "type": "STRING", 
                                    "enum": writ_types,
                                    "description": "The specific writ subtype as defined in the DB"
                                },
                                "damages_claimed": {"$ref": "#/$defs/currency"},
                                "damages_awarded": {"$ref": "#/$defs/currency"},
                                "costs": {"$ref": "#/$defs/currency"},
                                "pleading_summary": {"type": "STRING"}
                            }
                        },
                        "agents": {
                            "type": "ARRAY",
                            "items": {"$ref": "#/$defs/agent"}
                        }
                    },
                    "required": ["reference", "case_data", "agents"]
                }
            }
        }
        
        return schema

    def save_to_file(self, schema, filename="generated_schema.json"):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(schema, f, indent=4)
        print(f"Schema saved to {filename}")

if __name__ == "__main__":
    db_path = "CP40_Migrated.db"  # Adjust path as needed
    try:
        gen = SchemaGenerator(db_path)
        schema_obj = gen.generate()
        
        # Print a snippet to console
        print("\n--- EXTRACTED ENUM STATS ---")
        print(f"Roles found: {len(schema_obj['$defs']['agent']['properties']['roles']['items']['enum'])}")
        print(f"Occupations found: {len(schema_obj['$defs']['agent']['properties']['occupations']['items']['enum'])}")
        print(f"Writ Types found: {len(schema_obj['$defs']['caseRecord']['properties']['case_data']['properties']['writ_type']['enum'])}")
        
        gen.save_to_file(schema_obj)
        
        print("\nINSTRUCTIONS:")
        print("1. Open 'generated_schema.json'")
        print("2. Copy the content.")
        print("3. Paste it into your main workflow script, replacing the return value of 'get_final_index_schema()'.")
        
    except Exception as e:
        print(f"Error: {e}")