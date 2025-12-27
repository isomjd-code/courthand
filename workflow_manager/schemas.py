"""JSON schema helpers used by the workflow."""

from typing import Any, Dict


def get_diplomatic_schema() -> Dict[str, Any]:
    """
    Get JSON schema for Step 1 diplomatic transcription output.
    Includes Regex pattern enforcement for the strict 15th-c character set.

    Returns:
        JSON schema dictionary.
    """
    # Regex Explanation:
    # ^ and $ : Start and end of string (strict match)
    # A-Za-z  : Standard Latin alphabet
    # \.,;¶·  : Allowed punctuation
    # \s      : Whitespace
    # The rest: Specific medieval Unicode glyphs and combining diacritics
    # Note: \u0305 is the combining overline ( ̅ )
    
    allowed_pattern = r"^[A-Za-z\.,;¶·⁊ſħłđꝛꝝꝫꝑꝓꝙꝰꝭıþȝ÷\u0305ͬᵃᵉⁱᵒᵘ\s]+$"

    return {
        "type": "OBJECT",
        "properties": {
            "lines": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "id": {
                            "type": "STRING",
                            "description": "The line ID from the input HTR (e.g., L01)."
                        },
                        "transcription": {
                            "type": "STRING",
                            "description": "Diplomatic transcription. NO 'ñ', NO hyphens, NO precomposed macrons.",
                            "pattern": allowed_pattern
                        },
                    },
                    "required": ["id", "transcription"],
                },
            }
        },
        "required": ["lines"],
    }

def get_merged_diplomatic_schema() -> Dict[str, Any]:
    """
    Get JSON schema for Step 2a merged diplomatic output.

    Defines the structure for merged text with extracted surnames
    and place names (original and anglicized forms).

    Returns:
        JSON schema dictionary for merged diplomatic format.
    """
    return {
        "type": "OBJECT",
        "properties": {
            "merged_text": {
                "type": "STRING",
                "description": "The seamless, stitched diplomatic latin text.",
            },
            "surnames": {
                "type": "ARRAY",
                "items": {"type": "STRING"},
                "description": "List of all surnames found in the text.",
            },
            "place_names": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "original": {"type": "STRING"},
                        "anglicized": {"type": "STRING"},
                    },
                    "required": ["original", "anglicized"],
                },
                "description": "List of place names as tuples of (original, anglicized).",
            },
            "marginal_county": {
                "type": "OBJECT",
                "properties": {
                    "original": {"type": "STRING"},
                    "anglicized": {"type": "STRING"},
                },
                "required": ["original", "anglicized"],
                "description": "The name of the county as it appears in the margin",
            },            
        },
        "required": ["merged_text", "surnames", "place_names","marginal_county"],
    }


def get_final_index_schema() -> Dict[str, Any]:
    """
    Get JSON schema for Step 4 final structured extraction output.

    Defines the comprehensive schema for structured case data including:
    - Reference information (roll, rotulus, date, term, county)
    - Cases with parties, events, locations, legal details
    - Enumerated values for counties, case types, event types, roles, etc.

    Returns:
        JSON schema dictionary for final index format.
        This is the most complex schema, defining the complete structured output.
    """
    # Schema enhanced with constraints and descriptions from the Step 4 prompt
    return {
        "title": "CP40 Extraction",
        "type": "object",
        "properties": {
            "TblReference": {
                "type": "object",
                "description": "Reference information for the plea roll entry. All fields are required and should be populated from metadata when available.",
                "properties": {
                    "reference": {
                        "type": "string",
                        "description": "Roll reference constructed from Roll Number and rotulus (e.g., 'CP40-562 340'). Format: 'CP40-{RollNumber} {Rotulus}'"
                    },
                    "dateyear": {
                        "type": "integer",
                        "description": "Calendar year (e.g., 1427). Should be extracted from metadata 'Calendar Year' when available, or from text as fallback."
                    },
                    "term": {
                        "type": "string",
                        "description": "Court term. Must be extracted from metadata 'Term' when available, or from text. Must match one of the enum values.",
                        "enum": ["Michaelmas", "Hilary", "Easter", "Trinity"],
                    },
                    "County": {
                        "type": "string",
                        "description": "County name. Should be extracted from metadata 'County' when available (especially from marginal annotation), or from venue line in text as fallback. Must match one of the enum values.",
                        "enum": [
                            "Bristol",
                            "Southampton",
                            "York",
                            "Newcastle upon Tyne",
                            "Bedfordshire",
                            "Berkshire",
                            "Buckinghamshire",
                            "Cambridgeshire",
                            "Cheshire",
                            "Cornwall",
                            "Cumberland",
                            "Derbyshire",
                            "Devon",
                            "Dorset",
                            "Durham",
                            "Essex",
                            "Gloucestershire",
                            "Hampshire",
                            "Herefordshire",
                            "Hertfordshire",
                            "Huntingdonshire",
                            "Kent",
                            "Lancashire",
                            "Leicestershire",
                            "Lincolnshire",
                            "London",
                            "Middlesex",
                            "Norfolk",
                            "Northamptonshire",
                            "Northumberland",
                            "Nottinghamshire",
                            "Oxfordshire",
                            "Rutland",
                            "Shropshire",
                            "Somerset",
                            "Staffordshire",
                            "Suffolk",
                            "Surrey",
                            "Sussex",
                            "Warwickshire",
                            "Westmorland",
                            "Wiltshire",
                            "Worcestershire",
                            "Yorkshire",
                            "foreign",
                            "Wales",
                            "Kingston upon Hull",
                            "Norwich",
                            "Calais",
                            "Coventry",
                            "Lincoln",
                            "Westminster Palace",
                        ],
                    },
                },
                "required": ["reference", "dateyear", "term", "County"],
            },
            "Cases": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "TblCase": {
                            "type": "object",
                            "description": "Case information. DamClaimed and WritType are REQUIRED fields.",
                            "properties": {
                                "CaseRot": {
                                    "type": "string",
                                    "description": "Rotulus number for the case."
                                },
                                "County": {
                                    "type": "string",
                                    "description": "County name. Should match TblReference.County. Must match one of the enum values.",
                                    "enum": [
                                        "Bristol",
                                        "Southampton",
                                        "York",
                                        "Newcastle upon Tyne",
                                        "Bedfordshire",
                                        "Berkshire",
                                        "Buckinghamshire",
                                        "Cambridgeshire",
                                        "Cheshire",
                                        "Cornwall",
                                        "Cumberland",
                                        "Derbyshire",
                                        "Devon",
                                        "Dorset",
                                        "Durham",
                                        "Essex",
                                        "Gloucestershire",
                                        "Hampshire",
                                        "Herefordshire",
                                        "Hertfordshire",
                                        "Huntingdonshire",
                                        "Kent",
                                        "Lancashire",
                                        "Leicestershire",
                                        "Lincolnshire",
                                        "London",
                                        "Middlesex",
                                        "Norfolk",
                                        "Northamptonshire",
                                        "Northumberland",
                                        "Nottinghamshire",
                                        "Oxfordshire",
                                        "Rutland",
                                        "Shropshire",
                                        "Somerset",
                                        "Staffordshire",
                                        "Suffolk",
                                        "Surrey",
                                        "Sussex",
                                        "Warwickshire",
                                        "Westmorland",
                                        "Wiltshire",
                                        "Worcestershire",
                                        "Yorkshire",
                                        "foreign",
                                        "Wales",
                                        "Kingston upon Hull",
                                        "Norwich",
                                        "Calais",
                                        "Coventry",
                                        "Lincoln",
                                        "Westminster Palace",
                                    ],
                                },
                                "DamClaimed": {
                                    "type": "string",
                                    "description": "REQUIRED: Damages claimed by the plaintiff. Extract EXACT amount with original currency unit (e.g., '100s' for 100 shillings, '40 marks', '£10', '£10 5s 3d'). Preserve original format. Use empty string '' if not found after thorough search. CRITICAL: Distinguish between shillings (s.), marks (m. or 'marks'), and pounds (£ or li.)."
                                },
                                "DamAwarded": {
                                    "type": "string",
                                    "description": "Damages awarded by the court (if mentioned in judgment). Format same as DamClaimed."
                                },
                                "WritType": {
                                    "type": "string",
                                    "description": "REQUIRED: The form of action (legal category of the writ). Examples: 'Trespass', 'Debt', 'Account', 'Detinue', 'Covenant', 'Replevin', 'Waste', 'Dower'. Cannot be empty. Look for keywords like 'force and arms' (Trespass), 'plea that he render' (Debt), 'render reasonable account' (Account), 'unjustly detains' (Detinue)."
                                },
                                "CaseNotes": {
                                    "type": "string",
                                    "description": "Additional notes about the case."
                                },
                            },
                            "required": ["DamClaimed", "WritType"],
                        },
                        "TblCaseType": {
                            "type": "object",
                            "description": "Case type classification. CaseType is MANDATORY and must contain at least one entry.",
                            "properties": {
                                "CaseType": {
                                    "type": "array",
                                    "description": "REQUIRED: Array of case types. Must contain at least ONE case type. Analyze the NARRATIO (facts/plea section) to identify specific sub-categories. This represents the SPECIFIC facts of the case, NOT just the writ category. For example, a WritType of 'Trespass' may have CaseType of ['Assault'] or ['Housebreaking']. Extract ALL applicable case types.",
                                    "minItems": 1,
                                    "items": {
                                        "type": "string",
                                        "enum": [
                                            "Abduction",
                                            "Arbitration",
                                            "Assault",
                                            "Bond",
                                            "Breach of Statute",
                                            "Contract (not service/employment, or marriage)",
                                            "Debt",
                                            "Detention of goods",
                                            "Dower",
                                            "Embracery",
                                            "Housebreaking",
                                            "Imprisonment",
                                            "Loan",
                                            "Maintenance",
                                            "Negligence",
                                            "Real action  / rents / damage to real estate",
                                            "Reckoning of Account",
                                            "Safe keeping",
                                            "Sale of goods",
                                            "Surety of peace",
                                            "Theft",
                                            "Trespass (Chattels)",
                                            "Usurpation / abuse of rights",
                                            "Namium",
                                            "Trespass (Undefined)",
                                            "Taking of goods (de bonis asportatis)",
                                            "Contract (service/apprenticeship)",
                                            "Marriage contract",
                                        ],
                                    },
                                }
                            },
                        },
                        "TblEvents": {
                            "type": "array",
                            "description": "MANDATORY when events are mentioned in the text: Array of events. Must contain at least one entry when events like bonds, contracts, accounting sessions, payments, or property transfers are mentioned. Extract ALL events such as 'bond made on [Date] at [Place]', 'accounting at [Place] on [Date]', etc. If no events are mentioned in the text, this array may be empty.",
                            "items": {
                                "type": "object",
                                "description": "Event information. EventType is required.",
                                "properties": {
                                    "EventType": {
                                        "type": "string",
                                        "description": "REQUIRED: Type of event. Must match one of the enum values exactly. Common events: 'bond' (bond/obligation made), 'contract (not service/employment)' (contract/agreement), 'payment' (payment due/paid), 'accounting' (account/reckoning), 'sale of goods', 'loan', 'gift', 'property transfer', 'charter', 'will'.",
                                        "enum": [
                                            "abduction",
                                            "accounting",
                                            "arbitration",
                                            "arrest",
                                            "assault",
                                            "bond",
                                            "breach of statute",
                                            "payment",
                                            "breach of tenure",
                                            "charter",
                                            "contract (not service/employment)",
                                            "destruction of chattels",
                                            "negligence",
                                            "detention of goods",
                                            "disseisin",
                                            "good behaviour",
                                            "house-breaking",
                                            "imprisonment",
                                            "letters patent",
                                            "loan",
                                            "location of property",
                                            "maintenance",
                                            "marriage agreement",
                                            "property transfer",
                                            "recognizance",
                                            "release (from debt/obligation)",
                                            "rental agreement",
                                            "safe keeping",
                                            "sale of goods",
                                            "gift",
                                            "service/employment contract",
                                            "theft",
                                            "annuity",
                                            "trespass",
                                            "will",
                                            "writ",
                                            "namium",
                                            "taking of goods",
                                        ],
                                    },
                                    "EventDetails": {
                                        "type": "object",
                                        "properties": {
                                            "ValueAmount": {"type": "string"},
                                            "ValueDescription": {"type": "string"},
                                        },
                                    },
                                    "EventDate": {
                                        "type": "array",
                                        "description": "Array of dates associated with the event. Each date should include Date (ISO format) and DateType.",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "Date": {
                                                    "type": "string",
                                                    "description": "Date in ISO format (YYYY-MM-DD). Convert medieval feast dates to calendar equivalents (e.g., 'feast of St. Michael' → '1427-09-29'). Use regnal year and calendar year from metadata for accurate conversion.",
                                                    "pattern": "^\\d{4}-\\d{2}-\\d{2}$"
                                                },
                                                "DateType": {
                                                    "type": "string",
                                                    "description": "Type of date: 'initial' (when event was initiated), 'due' (when payment/action was due), 'occurred' (when event occurred), 'unknown' (date type unclear).",
                                                    "enum": ["initial", "due", "occurred", "unknown"],
                                                },
                                            },
                                        },
                                    },
                                    "LocationDetails": {
                                        "type": "object",
                                        "properties": {
                                            "SpecificPlace": {"type": "string"},
                                            "Parish": {"type": "string"},
                                            "Ward": {
                                                "type": "string",
                                                "enum": [
                                                    "Aldersgate Ward",
                                                    "Aldgate Ward",
                                                    "Bassishaw Ward",
                                                    "Billingsgate Ward",
                                                    "Bishopsgate Ward",
                                                    "Bread Street Ward",
                                                    "Bridge Ward",
                                                    "Broad Street Ward",
                                                    "Candlewick Street Ward",
                                                    "Castle Baynard Ward",
                                                    "Cheap Ward",
                                                    "Coleman Street Ward",
                                                    "Cordwainer Street Ward",
                                                    "Cornhill Ward",
                                                    "Cripplegate Ward",
                                                    "Dowgate Ward",
                                                    "Farringdon Ward Within",
                                                    "Farringdon Ward Without",
                                                    "Langbourn Ward",
                                                    "Lime Street Ward",
                                                    "Portsoken Ward",
                                                    "Queenhithe Ward",
                                                    "Tower Ward",
                                                    "Vintry Ward",
                                                    "Walbrook Ward",
                                                ],
                                            },
                                            "County": {
                                                "type": "string",
                                                "enum": [
                                                    "Bristol",
                                                    "Southampton",
                                                    "York",
                                                    "Newcastle upon Tyne",
                                                    "Bedfordshire",
                                                    "Berkshire",
                                                    "Buckinghamshire",
                                                    "Cambridgeshire",
                                                    "Cheshire",
                                                    "Cornwall",
                                                    "Cumberland",
                                                    "Derbyshire",
                                                    "Devon",
                                                    "Dorset",
                                                    "Durham",
                                                    "Essex",
                                                    "Gloucestershire",
                                                    "Hampshire",
                                                    "Herefordshire",
                                                    "Hertfordshire",
                                                    "Huntingdonshire",
                                                    "Kent",
                                                    "Lancashire",
                                                    "Leicestershire",
                                                    "Lincolnshire",
                                                    "London",
                                                    "Middlesex",
                                                    "Norfolk",
                                                    "Northamptonshire",
                                                    "Northumberland",
                                                    "Nottinghamshire",
                                                    "Oxfordshire",
                                                    "Rutland",
                                                    "Shropshire",
                                                    "Somerset",
                                                    "Staffordshire",
                                                    "Suffolk",
                                                    "Surrey",
                                                    "Sussex",
                                                    "Warwickshire",
                                                    "Westmorland",
                                                    "Wiltshire",
                                                    "Worcestershire",
                                                    "Yorkshire",
                                                    "foreign",
                                                    "Wales",
                                                    "Kingston upon Hull",
                                                    "Norwich",
                                                    "Calais",
                                                    "Coventry",
                                                    "Lincoln",
                                                    "Westminster Palace",
                                                ],
                                            },
                                            "Country": {"type": "string"},
                                        },
                                    },
                                },
                                "required": ["EventType"],
                            },
                        },
                        "Agents": {
                            "type": "array",
                            "description": "REQUIRED: Array of agents (people) involved in the case. Must contain at least one entry. Extract ALL people mentioned in the text. Every agent MUST have a TblAgentRole.role field populated.",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "description": "Agent (person) information. TblName and TblAgentRole are required. Every agent MUST have a role.",
                                "properties": {
                                    "TblName": {
                                        "type": "object",
                                        "description": "Name information. Surname is required. Anglicize names (Johannes → John, Henricus → Henry). Extract surnames EXACTLY as written. Validate against Latin text for accuracy.",
                                        "properties": {
                                            "Christian_name": {
                                                "type": "string",
                                                "description": "First/given name (anglicized). Examples: 'John' (from Johannes), 'Henry' (from Henricus), 'Roger' (from Rog'us - NOT Robert)."
                                            },
                                            "Surname": {
                                                "type": "string",
                                                "description": "REQUIRED: Family name. Extract EXACTLY as written. Do not substitute similar-sounding names. Examples: 'Sauvage' and 'Stannage' are DIFFERENT surnames."
                                            },
                                            "Suffix": {
                                                "type": "string",
                                                "description": "Name suffix (e.g., 'Jr.', 'Sr.', 'the Elder')."
                                            },
                                        },
                                        "required": ["Surname"],
                                    },
                                    "TblAgentRole": {
                                        "type": "object",
                                        "description": "REQUIRED: Agent role information. The role field is MANDATORY FOR EVERY AGENT. Cannot be omitted. Use 'Other' if no specific role fits, but you MUST assign a role.",
                                        "properties": {
                                            "role": {
                                                "type": "string",
                                                "description": "REQUIRED: Agent role. MANDATORY FOR EVERY AGENT. Must be one of the enum values. Identify role based on context: Plaintiff (brings case), Defendant (must answer), Debtor (owes money), Creditor (owed money), Attorney of plaintiff/defendant (represents party), Surety for defendant (guarantees appearance), Executor (executes will), Testator (made will), Accessory (assisted in crime/wrong), Witness, Juror, Justice, Clerk, etc. DO NOT leave any agent without a role.",
                                                "enum": [
                                                    "Accessory",
                                                    "Administrator",
                                                    "Arbitrator",
                                                    "Attorney of plaintiff",
                                                    "Attorney of defendant",
                                                    "Attorney of third party",
                                                    "Auditor",
                                                    "Chief justice",
                                                    "Clerk",
                                                    "Creditor",
                                                    "Debtor",
                                                    "Defendant",
                                                    "Essoin of defendant",
                                                    "Essoin of plaintiff",
                                                    "Executor",
                                                    "Intestator",
                                                    "Juror",
                                                    "Justice",
                                                    "Official",
                                                    "Other",
                                                    "Plaintiff",
                                                    "Surety for defendant",
                                                    "Surety of law (compurgator)",
                                                    "Surety other",
                                                    "Surety of Plaintiff",
                                                    "Testator",
                                                    "Witness",
                                                ],
                                            }
                                        },
                                        "required": ["role"],
                                    },
                                    "TblAgentStatus": {
                                        "type": "object",
                                        "required": ["AgentStatus"],
                                        "properties": {
                                            "AgentStatus": {
                                                "type": "string",
                                                "description": "Social status/Rank.",
                                                "enum": [
                                                    "burgess",
                                                    "alderman",
                                                    "exalderman",
                                                    "bailiff",
                                                    "exbailiff",
                                                    "companyofficer",
                                                    "churchwarden",
                                                    "parishclergy",
                                                    "citizen",
                                                    "companyfree",
                                                    "apprentice",
                                                    "servant",
                                                    "preacher",
                                                    "mr",
                                                    "sir",
                                                    "master",
                                                    "dr",
                                                    "parishioner",
                                                    "householder",
                                                    "gentleman",
                                                    "esquire",
                                                    "knight",
                                                    "knight bar",
                                                    "lordeccles",
                                                    "lordsec",
                                                    "bishop",
                                                    "archbishop",
                                                    "earl",
                                                    "duke",
                                                    "lady",
                                                    "dame",
                                                    "royalofficer",
                                                    "civicofficer",
                                                    "chamberlainLondon",
                                                    "mistress",
                                                    "freeco",
                                                    "father",
                                                    "sergeant",
                                                    "illegitimate",
                                                    "mother",
                                                    "clerk",
                                                    "poor",
                                                    "parishofficer",
                                                    "lodger",
                                                    "captain",
                                                    "madam",
                                                    "major",
                                                    "sister",
                                                    "gentlewoman",
                                                    "overseas",
                                                    "ltcol",
                                                    "deputy",
                                                    "constable",
                                                    "black",
                                                    "journeyman",
                                                    "inmate",
                                                    "parishchild",
                                                    "pensioner",
                                                    "drphys",
                                                    "drlaw",
                                                    "officer",
                                                    "foundling",
                                                    "papist",
                                                    "roman cathollic",
                                                    "honourable",
                                                    "beadlelist",
                                                    "RightHon",
                                                    "LordMayor",
                                                    "stranger",
                                                    "mp",
                                                    "king",
                                                    "goodwife",
                                                    "worshipful",
                                                    "baron",
                                                    "colonel",
                                                    "justice",
                                                    "yeoman",
                                                    "widow",
                                                    "wife",
                                                    "Bachelor of Law",
                                                    "canon",
                                                    "dean",
                                                    "keeper of spiritualities",
                                                    "outlaw",
                                                    "franklin",
                                                ],
                                            }
                                        },
                                    },
                                    "TblAgent": {
                                        "type": "object",
                                        "properties": {
                                            "AgentGender": {
                                                "type": "string",
                                                "enum": ["m", "f", "u", "mixed"],
                                            },
                                            "Occupation": {
                                                "type": "string",
                                                "description": "REQUIRED when mentioned in text: Occupation or status. Extract EXACTLY as written (e.g., 'citizen and mercer', 'husbandman', 'prior', 'citizen'). Look for occupational terms: trades (mercer, skinner, goldsmith), agricultural (husbandman, yeoman), clerical (prior, dean, bishop, clerk), legal (attorney, serjeant), status (citizen, esquire, knight, gentleman, merchant). If multiple occupations/statuses mentioned, extract all. If no occupation mentioned, may be null/empty."
                                            },
                                            "Relationships": {
                                                "type": "string",
                                                "description": "Relationships to other people (e.g., 'son of', 'wife of', 'executor of')."
                                            },
                                            "InstitutionName": {
                                                "type": "string",
                                                "description": "Name of institution associated with the agent (e.g., monastery, company, guild)."
                                            },
                                            "LocationDetails": {
                                                "type": "object",
                                                "description": "REQUIRED when mentioned in text: Location. Look for patterns: 'X de [Place]' (Latin), 'X of [Place]' (English), 'X, [Place]'. Extract when location appears in venue/identification section. Use historically accurate place name normalization. If no location mentioned, may be null/empty.",
                                                "properties": {
                                                    "SpecificPlace": {
                                                        "type": "string",
                                                        "description": "Specific place name (e.g., 'Southampton', 'London', 'Sutton'). Use historically accurate normalization (e.g., 'Sutht'' → 'Sutton', 'Lond'' → 'London')."
                                                    },
                                                    "Parish": {
                                                        "type": "string",
                                                        "description": "Parish name if mentioned (e.g., 'parish of St. John Walbrook')."
                                                    },
                                                    "Ward": {
                                                        "type": "string",
                                                        "description": "Ward name (for London locations). Must match one of the enum values.",
                                                        "enum": [
                                                            "Aldersgate Ward",
                                                            "Aldgate Ward",
                                                            "Bassishaw Ward",
                                                            "Billingsgate Ward",
                                                            "Bishopsgate Ward",
                                                            "Bread Street Ward",
                                                            "Bridge Ward",
                                                            "Broad Street Ward",
                                                            "Candlewick Street Ward",
                                                            "Castle Baynard Ward",
                                                            "Cheap Ward",
                                                            "Coleman Street Ward",
                                                            "Cordwainer Street Ward",
                                                            "Cornhill Ward",
                                                            "Cripplegate Ward",
                                                            "Dowgate Ward",
                                                            "Farringdon Ward Within",
                                                            "Farringdon Ward Without",
                                                            "Langbourn Ward",
                                                            "Lime Street Ward",
                                                            "Portsoken Ward",
                                                            "Queenhithe Ward",
                                                            "Tower Ward",
                                                            "Vintry Ward",
                                                            "Walbrook Ward",
                                                        ],
                                                    },
                                                    "County": {
                                                        "type": "string",
                                                        "enum": [
                                                            "Bristol",
                                                            "Southampton",
                                                            "York",
                                                            "Newcastle upon Tyne",
                                                            "Bedfordshire",
                                                            "Berkshire",
                                                            "Buckinghamshire",
                                                            "Cambridgeshire",
                                                            "Cheshire",
                                                            "Cornwall",
                                                            "Cumberland",
                                                            "Derbyshire",
                                                            "Devon",
                                                            "Dorset",
                                                            "Durham",
                                                            "Essex",
                                                            "Gloucestershire",
                                                            "Hampshire",
                                                            "Herefordshire",
                                                            "Hertfordshire",
                                                            "Huntingdonshire",
                                                            "Kent",
                                                            "Lancashire",
                                                            "Leicestershire",
                                                            "Lincolnshire",
                                                            "London",
                                                            "Middlesex",
                                                            "Norfolk",
                                                            "Northamptonshire",
                                                            "Northumberland",
                                                            "Nottinghamshire",
                                                            "Oxfordshire",
                                                            "Rutland",
                                                            "Shropshire",
                                                            "Somerset",
                                                            "Staffordshire",
                                                            "Suffolk",
                                                            "Surrey",
                                                            "Sussex",
                                                            "Warwickshire",
                                                            "Westmorland",
                                                            "Wiltshire",
                                                            "Worcestershire",
                                                            "Yorkshire",
                                                            "foreign",
                                                            "Wales",
                                                            "Kingston upon Hull",
                                                            "Norwich",
                                                            "Calais",
                                                            "Coventry",
                                                            "Lincoln",
                                                            "Westminster Palace",
                                                        ],
                                                    },
                                                    "Country": {
                                                        "type": "string",
                                                        "description": "Country name (usually 'England' for most cases, or as mentioned in text)."
                                                    },
                                                },
                                            },
                                        },
                                    },
                                },
                                "required": ["TblName", "TblAgentRole"],
                            },
                        },
                        "TblPleadings": {
                            "type": "array",
                            "description": "MANDATORY: Array of pleading steps. Must contain at least one entry. Each entry must be ONE SENTENCE describing a pleading step (Count/Narratio, Defense/Bar, Replication, or Issue). Extract ALL legal arguments from the text. Do not summarize - break down complex sentences into separate entries.",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "description": "Pleading step. Each PleadingText must be a single, complete sentence.",
                                "properties": {
                                    "PleadingText": {
                                        "type": "string",
                                        "description": "REQUIRED: One sentence describing a pleading step. Examples: 'The plaintiff claims bond for 40 marks made on [Date]' (Count), 'The defendant says he owes nothing' (Defense), 'The plaintiff puts himself on the country' (Issue). Each entry must be ONE SENTENCE."
                                    }
                                },
                                "required": ["PleadingText"]
                            },
                        },
                        "TblPostea": {
                            "type": "array",
                            "description": "MANDATORY: Array of postea events (procedural events after pleadings). Must contain at least one entry. Extract ALL postea events: Sheriff returns, adjournments, defaults, judgments. Look for keywords: 'Afterwards', 'Postea', 'At which day', 'Sheriff', 'It is considered', 'Consideratum est'. Read the ENTIRE document - do NOT stop at end of pleadings.",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "description": "Postea event. Each PosteaText must be a single, complete sentence.",
                                "properties": {
                                    "PosteaText": {
                                        "type": "string",
                                        "description": "REQUIRED: One sentence describing a postea event. Examples: 'At which day both parties appeared', 'The sheriff did not send the writ', 'It is considered that [Plaintiff] recover [amount]'. Each entry must be ONE SENTENCE."
                                    },
                                    "Date": {
                                        "type": "string",
                                        "description": "Date in ISO format (YYYY-MM-DD) when mentioned. Convert medieval feast dates to calendar equivalents.",
                                        "pattern": "^\\d{4}-\\d{2}-\\d{2}$"
                                    },
                                },
                                "required": ["PosteaText"]
                            },
                        },
                    },
                    "required": ["TblCase", "TblPleadings", "TblPostea"],
                },
            },
        },
        "required": ["Cases"],
        "description": "CP40 Plea Roll Extraction Schema. This schema defines the complete structured output for Court of Common Pleas records. All mandatory fields must be populated. Arrays marked as MANDATORY must contain at least one entry when the relevant information is present in the source text."
    }

