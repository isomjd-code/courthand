# Database Schema Documentation

This document describes the schemas for two CP40-related databases:
- `cp40_records.db` - Database for scraped CP40 records
- `cp40_database_new.sqlite` - Database for processed legal case data

---

## cp40_records.db

This database stores scraped CP40 records with normalized entities (persons, places, surnames) and tracks scraping progress.

### Core Tables

#### `entries`
Main table storing CP40 record entries.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER | PRIMARY KEY, AUTOINCREMENT | Unique entry identifier |
| `roll_reference` | TEXT | NOT NULL | Reference to the roll document |
| `raw_text` | TEXT | NOT NULL | Full raw text of the entry |
| `raw_text_hash` | TEXT | UNIQUE, NOT NULL | SHA256 hash for deduplication |
| `county` | TEXT | | County name |
| `year` | INTEGER | | Year of the record |
| `created_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | Timestamp when entry was created |

**Indexes:**
- `idx_entries_year` on `year`
- `idx_entries_county` on `county`
- `idx_entries_roll_ref` on `roll_reference`

#### `persons`
Normalized person names (many-to-many with entries).

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER | PRIMARY KEY, AUTOINCREMENT | Unique person identifier |
| `name` | TEXT | NOT NULL, UNIQUE | Full person name |

**Indexes:**
- `idx_persons_name` on `name`

#### `entry_persons`
Junction table linking entries to persons (many-to-many relationship).

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `entry_id` | INTEGER | PRIMARY KEY, NOT NULL | Foreign key to `entries.id` |
| `person_id` | INTEGER | PRIMARY KEY, NOT NULL | Foreign key to `persons.id` |

**Foreign Keys:**
- `entry_id` → `entries(id)` ON DELETE CASCADE
- `person_id` → `persons(id)` ON DELETE CASCADE

#### `places`
Normalized place names (many-to-many with entries).

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER | PRIMARY KEY, AUTOINCREMENT | Unique place identifier |
| `name` | TEXT | NOT NULL, UNIQUE | Place name |
| `processed_at` | TIMESTAMP | | Timestamp when place was processed |
| `frequency` | INTEGER | DEFAULT 0 | Frequency count |

**Indexes:**
- `idx_places_name` on `name`

#### `entry_places`
Junction table linking entries to places (many-to-many relationship).

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `entry_id` | INTEGER | PRIMARY KEY, NOT NULL | Foreign key to `entries.id` |
| `place_id` | INTEGER | PRIMARY KEY, NOT NULL | Foreign key to `places.id` |

**Foreign Keys:**
- `entry_id` → `entries(id)` ON DELETE CASCADE
- `place_id` → `places(id)` ON DELETE CASCADE

#### `links`
Document/image links associated with entries (one-to-many: entry has multiple links).

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER | PRIMARY KEY, AUTOINCREMENT | Unique link identifier |
| `entry_id` | INTEGER | NOT NULL | Foreign key to `entries.id` |
| `url` | TEXT | NOT NULL | URL to the document/image |
| `link_text` | TEXT | | Descriptive text for the link |

**Foreign Keys:**
- `entry_id` → `entries(id)` ON DELETE CASCADE

### Surname Normalization

#### `surnames`
Normalized surnames extracted from person names (last word).

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER | PRIMARY KEY, AUTOINCREMENT | Unique surname identifier |
| `surname` | TEXT | NOT NULL, UNIQUE | Surname text |

**Indexes:**
- `idx_surnames_surname` on `surname`

#### `person_surnames`
Junction table linking persons to surnames.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `person_id` | INTEGER | PRIMARY KEY, NOT NULL | Foreign key to `persons.id` |
| `surname_id` | INTEGER | PRIMARY KEY, NOT NULL | Foreign key to `surnames.id` |

**Foreign Keys:**
- `person_id` → `persons(id)` ON DELETE CASCADE
- `surname_id` → `surnames(id)` ON DELETE CASCADE

### Scraping Progress Tracking

#### `scrape_jobs`
Tracks year-level scraping progress (for resumability).

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER | PRIMARY KEY, AUTOINCREMENT | Unique job identifier |
| `year` | INTEGER | NOT NULL | Year being scraped |
| `county` | TEXT | | County filter (NULL for all counties) |
| `status` | TEXT | DEFAULT 'pending' | Job status: pending, in_progress, completed, failed |
| `results_count` | INTEGER | DEFAULT 0 | Total results found |
| `new_entries_count` | INTEGER | DEFAULT 0 | New entries added |
| `duplicate_count` | INTEGER | DEFAULT 0 | Duplicate entries found |
| `error_message` | TEXT | | Error message if failed |
| `started_at` | TIMESTAMP | | When job started |
| `completed_at` | TIMESTAMP | | When job completed |

**Constraints:**
- UNIQUE(year, county)

**Indexes:**
- `idx_scrape_jobs_status` on `status`
- `idx_scrape_jobs_year` on `year`

#### `scrape_prefix_jobs`
Tracks prefix-level scraping progress (year + surname prefix combinations like a*, b*, c*, etc.).

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER | PRIMARY KEY, AUTOINCREMENT | Unique job identifier |
| `year` | INTEGER | NOT NULL | Year being scraped |
| `prefix` | TEXT | NOT NULL | Surname prefix (single letter) |
| `status` | TEXT | DEFAULT 'pending' | Job status: pending, in_progress, completed, failed |
| `results_count` | INTEGER | DEFAULT 0 | Total results found |
| `new_entries_count` | INTEGER | DEFAULT 0 | New entries added |
| `duplicate_count` | INTEGER | DEFAULT 0 | Duplicate entries found |
| `error_message` | TEXT | | Error message if failed |
| `started_at` | TIMESTAMP | | When job started |
| `completed_at` | TIMESTAMP | | When job completed |

**Constraints:**
- UNIQUE(year, prefix)

**Indexes:**
- `idx_scrape_prefix_jobs_year` on `year`
- `idx_scrape_prefix_jobs_status` on `status`

### Latin Form Processing

#### `forenames`
English/anglicized forenames for Latin form processing.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER | PRIMARY KEY, AUTOINCREMENT | Unique forename identifier |
| `english_name` | TEXT | NOT NULL, UNIQUE | English/anglicized forename |
| `frequency` | INTEGER | DEFAULT 0 | Frequency count |
| `gender` | TEXT | | Gender: 'm', 'f', or NULL |
| `processed_at` | TIMESTAMP | | When forename was processed |
| `created_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | When record was created |

**Indexes:**
- `idx_forenames_english` on `english_name`
- `idx_forenames_frequency` on `frequency DESC`

#### `forename_latin_forms`
Latin forms (declined and abbreviated) for forenames.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER | PRIMARY KEY, AUTOINCREMENT | Unique form identifier |
| `forename_id` | INTEGER | NOT NULL | Foreign key to `forenames.id` |
| `case_name` | TEXT | NOT NULL | Latin case: nominative, genitive, dative, accusative, ablative |
| `latin_full` | TEXT | NOT NULL | Full Latin form (e.g., "Johannes") |
| `latin_abbreviated` | TEXT | NOT NULL | Abbreviated form (e.g., "Joh'es") |
| `is_primary` | BOOLEAN | DEFAULT 0 | Whether this is the primary/canonical form |
| `notes` | TEXT | | Special notes about this form |
| `created_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | When record was created |
| `normalized_form` | TEXT | | Normalized form for lookup (I/J, U/V normalized) |
| `variant_type` | TEXT | DEFAULT 'primary' | Variant type: primary, ij_variant, uv_variant, abbrev_variant, etc. |

**Constraints:**
- UNIQUE(forename_id, case_name, latin_abbreviated)

**Foreign Keys:**
- `forename_id` → `forenames(id)` ON DELETE CASCADE

**Indexes:**
- `idx_latin_forms_forename` on `forename_id`
- `idx_latin_forms_case` on `case_name`
- `idx_latin_forms_abbreviated` on `latin_abbreviated`

#### `forename_processing_jobs`
Tracks API call progress for forename Latin form generation.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER | PRIMARY KEY, AUTOINCREMENT | Unique job identifier |
| `forename_id` | INTEGER | NOT NULL | Foreign key to `forenames.id` |
| `status` | TEXT | DEFAULT 'pending' | Status: pending, in_progress, completed, failed |
| `error_message` | TEXT | | Error message if failed |
| `api_response` | TEXT | | Raw API response for debugging |
| `started_at` | TIMESTAMP | | When processing started |
| `completed_at` | TIMESTAMP | | When processing completed |
| `variants_generated` | INTEGER | DEFAULT 0 | Number of variants generated |

**Constraints:**
- UNIQUE(forename_id)

**Foreign Keys:**
- `forename_id` → `forenames(id)` ON DELETE CASCADE

**Indexes:**
- `idx_processing_status` on `status`

#### `place_latin_forms`
Latin forms for place names.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER | PRIMARY KEY, AUTOINCREMENT | Unique form identifier |
| `place_id` | INTEGER | NOT NULL | Foreign key to `places.id` |
| `latin_full` | TEXT | NOT NULL | Full Latin form (e.g., "Londinium") |
| `latin_abbreviated` | TEXT | NOT NULL | Abbreviated form (e.g., "Lond'ium") |
| `is_primary` | BOOLEAN | DEFAULT 0 | Whether this is the primary/canonical form |
| `normalized_form` | TEXT | | Normalized form for lookup (I/J, U/V normalized) |
| `variant_type` | TEXT | DEFAULT 'primary' | Variant type: primary, ij_variant, uv_variant, abbrev_variant, etc. |
| `notes` | TEXT | | Special notes about this form |
| `created_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | When record was created |

**Constraints:**
- UNIQUE(place_id, latin_abbreviated)

**Foreign Keys:**
- `place_id` → `places(id)` ON DELETE CASCADE

**Indexes:**
- `idx_latin_forms_place` on `place_id`
- `idx_latin_forms_abbreviated` on `latin_abbreviated`
- `idx_latin_forms_normalized` on `normalized_form`

#### `place_processing_jobs`
Tracks API call progress for place Latin form generation.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER | PRIMARY KEY, AUTOINCREMENT | Unique job identifier |
| `place_id` | INTEGER | NOT NULL | Foreign key to `places.id` |
| `status` | TEXT | DEFAULT 'pending' | Status: pending, in_progress, completed, failed |
| `error_message` | TEXT | | Error message if failed |
| `api_response` | TEXT | | Raw API response for debugging |
| `started_at` | TIMESTAMP | | When processing started |
| `completed_at` | TIMESTAMP | | When processing completed |
| `variants_generated` | INTEGER | DEFAULT 0 | Number of variants generated |

**Constraints:**
- UNIQUE(place_id)

**Foreign Keys:**
- `place_id` → `places(id)` ON DELETE CASCADE

**Indexes:**
- `idx_processing_status` on `status`

---

## cp40_database_new.sqlite

This database stores processed legal case data with detailed information about cases, events, pleadings, agents, and related entities. It appears to be a more structured database for storing analyzed and processed CP40 case information.

### Lookup Tables

#### `LookUpCaseType`
Case type lookup table.

| Column | Type | Description |
|--------|------|-------------|
| `CaseTypeID` | | Case type identifier |
| `A-Z Case Type` | | Case type name |
| `CaseTypeNotes` | | Notes about the case type |
| `OutAZCaseType` | | Output case type |
| `s_ColLineage` | | Column lineage |
| `s_Generation` | | Generation tracking |
| `s_GUID` | | Globally unique identifier |
| `s_Lineage` | | Lineage tracking |

#### `LookUpCounty`
County lookup table.

| Column | Type | Description |
|--------|------|-------------|
| `countyid` | | County identifier |
| `county` | | County name |
| `s_ColLineage` | | Column lineage |
| `s_Generation` | | Generation tracking |
| `s_GUID` | | Globally unique identifier |
| `s_Lineage` | | Lineage tracking |

#### `LookUpDocumentLanguage`
Document language lookup table.

| Column | Type | Description |
|--------|------|-------------|
| `languageid` | | Language identifier |
| `language` | | Language name |
| `s_ColLineage` | | Column lineage |
| `s_Generation` | | Generation tracking |
| `s_GUID` | | Globally unique identifier |
| `s_Lineage` | | Lineage tracking |

#### `LookUpEventType`
Event type lookup table.

| Column | Type | Description |
|--------|------|-------------|
| `EventID` | | Event identifier |
| `EventType` | | Event type name |
| `EventType notes` | | Notes about the event type |
| `s_ColLineage` | | Column lineage |
| `s_Generation` | | Generation tracking |
| `s_GUID` | | Globally unique identifier |
| `s_Lineage` | | Lineage tracking |

#### `LookUpFeast`
Feast date lookup table.

| Column | Type | Description |
|--------|------|-------------|
| `FeastID` | | Feast identifier |
| `Feastname` | | Feast name |
| `Feastdate` | | Feast date |
| `OutFeastName` | | Output feast name |
| `s_ColLineage` | | Column lineage |
| `s_Generation` | | Generation tracking |
| `s_GUID` | | Globally unique identifier |
| `s_Lineage` | | Lineage tracking |

#### `LookUpForenames`
Forename lookup table.

| Column | Type | Description |
|--------|------|-------------|
| `forenameid` | | Forename identifier |
| `forename` | | Forename text |
| `s_ColLineage` | | Column lineage |
| `s_Generation` | | Generation tracking |
| `s_GUID` | | Globally unique identifier |
| `s_Lineage` | | Lineage tracking |

#### `LookUpGoods`
Goods type lookup table.

| Column | Type | Description |
|--------|------|-------------|
| `GoodsTypeID` | | Goods type identifier |
| `GoodsType` | | Goods type name |
| `GoodsTypeDescription` | | Description of goods type |
| `s_ColLineage` | | Column lineage |
| `s_Generation` | | Generation tracking |
| `s_GUID` | | Globally unique identifier |
| `s_Lineage` | | Lineage tracking |

#### `LookUpIndividualRole`
Individual role lookup table.

| Column | Type | Description |
|--------|------|-------------|
| `individualroleid` | | Role identifier |
| `roleabbrev` | | Role abbreviation |
| `role` | | Full role name |
| `s_ColLineage` | | Column lineage |
| `s_Generation` | | Generation tracking |
| `s_GUID` | | Globally unique identifier |
| `s_Lineage` | | Lineage tracking |

#### `LookUpInstitutionType`
Institution type lookup table.

| Column | Type | Description |
|--------|------|-------------|
| `institutiontypeid` | | Institution type identifier |
| `institutiontype` | | Institution type name |
| `institutiontypenotes` | | Notes about institution type |
| `s_ColLineage` | | Column lineage |
| `s_Generation` | | Generation tracking |
| `s_GUID` | | Globally unique identifier |
| `s_Lineage` | | Lineage tracking |

#### `LookUpParish`
Parish lookup table.

| Column | Type | Description |
|--------|------|-------------|
| `parishid` | | Parish identifier |
| `parish` | | Parish name |
| `s_ColLineage` | | Column lineage |
| `s_Generation` | | Generation tracking |
| `s_GUID` | | Globally unique identifier |
| `s_Lineage` | | Lineage tracking |

#### `LookUpRelat`
Relationship lookup table.

| Column | Type | Description |
|--------|------|-------------|
| `relatid` | | Relationship identifier |
| `relat` | | Relationship abbreviation |
| `relatdesc` | | Relationship description |
| `s_ColLineage` | | Column lineage |
| `s_Generation` | | Generation tracking |
| `s_GUID` | | Globally unique identifier |
| `s_Lineage` | | Lineage tracking |

#### `LookUpSheriffsCourtEntryTypes`
Sheriff's court entry types lookup table.

| Column | Type | Description |
|--------|------|-------------|
| `id` | | Entry type identifier |
| `Events` | | Event type |
| `s_ColLineage` | | Column lineage |
| `s_Generation` | | Generation tracking |
| `s_Lineage` | | Lineage tracking |

#### `LookUpStatus`
Status lookup table.

| Column | Type | Description |
|--------|------|-------------|
| `statusid` | | Status identifier |
| `status` | | Status name |
| `statusdescription` | | Status description |
| `s_ColLineage` | | Column lineage |
| `s_Generation` | | Generation tracking |
| `s_GUID` | | Globally unique identifier |
| `s_Lineage` | | Lineage tracking |

#### `LookUpWard`
Ward lookup table.

| Column | Type | Description |
|--------|------|-------------|
| `wardid` | | Ward identifier |
| `ward` | | Ward name |
| `s_ColLineage` | | Column lineage |
| `s_Generation` | | Generation tracking |
| `s_GUID` | | Globally unique identifier |
| `s_Lineage` | | Lineage tracking |

#### `LookUpWritType(sub-type)`
Writ type sub-type lookup table.

| Column | Type | Description |
|--------|------|-------------|
| `Writ Type (sub-type) ID` | | Writ sub-type identifier |
| `Writ Type (sub-type)` | | Writ sub-type name |
| `OutWritTypeSubType` | | Output writ sub-type |
| `s_ColLineage` | | Column lineage |
| `s_Generation` | | Generation tracking |
| `s_GUID` | | Globally unique identifier |
| `s_Lineage` | | Lineage tracking |

#### `LookupAgentRole`
Agent role lookup table.

| Column | Type | Description |
|--------|------|-------------|
| `LookupAgentRoleID` | | Agent role identifier |
| `role` | | Role name |
| `DisplayRole` | | Display role name |
| `s_ColLineage` | | Column lineage |
| `s_Generation` | | Generation tracking |
| `s_GUID` | | Globally unique identifier |
| `s_Lineage` | | Lineage tracking |

### Core Data Tables

#### `TblCase`
Main case information table.

| Column | Type | Description |
|--------|------|-------------|
| `DocID` | | Document identifier |
| `CaseID` | | Case identifier |
| `County` | | County name |
| `Sc_ County` | | Scraped county |
| `CaseRot` | | Case rotulet reference |
| `DamClaimed` | | Damages claimed |
| `DamAward` | | Damages awarded |
| `Costs` | | Costs |
| `CaseNotes` | | Case notes |
| `Jury` | | Jury information |
| `Default` | | Default information |
| `Assize` | | Assize information |
| `Law` | | Law information |
| `Admits` | | Admissions |
| `Problem case` | | Problem case flag |
| `value of suit` | | Value of suit |
| `value in pence` | | Value in pence |
| `Essoin Only` | | Essoin only flag |
| `Day Only` | | Day only flag |
| `Gen_CaseNotes` | | Generated case notes |
| `s_ColLineage` | | Column lineage |
| `s_Generation` | | Generation tracking |
| `s_GUID` | | Globally unique identifier |
| `s_Lineage` | | Lineage tracking |

#### `TblCaseType`
Case type associations.

| Column | Type | Description |
|--------|------|-------------|
| `CaseID` | | Case identifier |
| `CaseType` | | Case type |
| `CaseTypeNotes` | | Case type notes |
| `Gen_CaseTypeNotes` | | Generated case type notes |
| `s_ColLineage` | | Column lineage |
| `s_Generation` | | Generation tracking |
| `s_GUID` | | Globally unique identifier |
| `s_Lineage` | | Lineage tracking |

#### `TblReference`
Document reference information.

| Column | Type | Description |
|--------|------|-------------|
| `docid` | | Document identifier |
| `reference` | | Document reference |
| `dateterm` | | Date term |
| `dateyear` | | Date year |
| `language` | | Document language |
| `notes` | | Notes |
| `Number of rotulets` | | Number of rotulets |
| `calculation date` | | Calculation date |
| `Record Series` | | Record series |
| `Gen_notes` | | Generated notes |
| `s_ColLineage` | | Column lineage |
| `s_Generation` | | Generation tracking |
| `s_GUID` | | Globally unique identifier |
| `s_Lineage` | | Lineage tracking |

### Event and Date Tables

#### `TblEvent`
Event information for cases.

| Column | Type | Description |
|--------|------|-------------|
| `CaseID` | | Case identifier |
| `EventID` | | Event identifier |
| `EventPlaceMs` | | Event place (manuscript) |
| `EventPlaceParish` | | Event place parish |
| `EventPlaceWard` | | Event place ward |
| `EventPlaceCounty` | | Event place county |
| `EventPlaceCountry` | | Event place country |
| `EventNotes` | | Event notes |
| `value` | | Event value |
| `value in approximate shillings` | | Value in approximate shillings |
| `value in pence` | | Value in pence |
| `EventPleadingCode` | | Event pleading code |
| `Gen_EventNotes` | | Generated event notes |
| `s_ColLineage` | | Column lineage |
| `s_Generation` | | Generation tracking |
| `s_GUID` | | Globally unique identifier |
| `s_Lineage` | | Lineage tracking |

#### `TblEventType`
Event type associations.

| Column | Type | Description |
|--------|------|-------------|
| `EventID` | | Event identifier |
| `EventType` | | Event type |
| `EventTypeNotes` | | Event type notes |
| `s_ColLineage` | | Column lineage |
| `s_Generation` | | Generation tracking |
| `s_GUID` | | Globally unique identifier |
| `s_Lineage` | | Lineage tracking |

#### `TblInitialDates`
Initial dates for events.

| Column | Type | Description |
|--------|------|-------------|
| `EventID` | | Event identifier |
| `InitialDate` | | Initial date |
| `s_ColLineage` | | Column lineage |
| `s_Generation` | | Generation tracking |
| `s_GUID` | | Globally unique identifier |
| `s_Lineage` | | Lineage tracking |

#### `TblDueDates`
Due dates for events.

| Column | Type | Description |
|--------|------|-------------|
| `EventID` | | Event identifier |
| `Feast` | | Feast reference |
| `DueDate` | | Due date |
| `Boguskey` | | Bogus key |
| `interval` | | Interval |
| `s_ColLineage` | | Column lineage |
| `s_Generation` | | Generation tracking |
| `s_GUID` | | Globally unique identifier |
| `s_Lineage` | | Lineage tracking |

### Person and Name Tables

#### `TblName`
Person name information.

| Column | Type | Description |
|--------|------|-------------|
| `pid` | | Person identifier |
| `AutonumberKey` | | Autonumber key |
| `Christian name` | | Christian/forename |
| `Surname` | | Surname |
| `Suffix` | | Name suffix |
| `NameCaseID` | | Name case identifier |
| `s_ColLineage` | | Column lineage |
| `s_Generation` | | Generation tracking |
| `s_GUID` | | Globally unique identifier |
| `s_Lineage` | | Lineage tracking |

#### `TblCountyandLocation`
County and location associations for persons.

| Column | Type | Description |
|--------|------|-------------|
| `Pid` | | Person identifier |
| `County` | | County name |
| `Location` | | Location name |
| `Nuper` | | Nuper flag |
| `dummy joint key` | | Dummy joint key |
| `s_ColLineage` | | Column lineage |
| `s_Generation` | | Generation tracking |
| `s_GUID` | | Globally unique identifier |
| `s_Lineage` | | Lineage tracking |

#### `TblOccupation`
Occupation information for persons.

| Column | Type | Description |
|--------|------|-------------|
| `pid` | | Person identifier |
| `AgentOccupation` | | Agent occupation |
| `s_ColLineage` | | Column lineage |
| `s_Generation` | | Generation tracking |
| `s_GUID` | | Globally unique identifier |
| `s_Lineage` | | Lineage tracking |

#### `TblRelats`
Relationships between persons.

| Column | Type | Description |
|--------|------|-------------|
| `RelatId` | | Relationship identifier |
| `Pid` | | Person identifier |
| `RelatedTo` | | Related person identifier |
| `Relationship` | | Relationship type |
| `RelatNotes` | | Relationship notes |
| `RelatCaseID` | | Related case identifier |
| `s_ColLineage` | | Column lineage |
| `s_Generation` | | Generation tracking |
| `s_GUID` | | Globally unique identifier |
| `s_Lineage` | | Lineage tracking |

### Agent Tables

#### `TblAgent`
Agent information.

| Column | Type | Description |
|--------|------|-------------|
| `CaseID` | | Case identifier |
| `pid` | | Person identifier |
| `InstitutionName` | | Institution name |
| `AgentGender` | | Agent gender |
| `AgentLocation` | | Agent location |
| `AgentParish` | | Agent parish |
| `AgentWard` | | Agent ward |
| `AgentCountry` | | Agent country |
| `AgentNotes` | | Agent notes |
| `Agent Deceased` | | Agent deceased flag |
| `Gen_AgentNotes` | | Generated agent notes |
| `s_ColLineage` | | Column lineage |
| `s_Generation` | | Generation tracking |
| `s_GUID` | | Globally unique identifier |
| `s_Lineage` | | Lineage tracking |

#### `TblAgentRole`
Agent role associations.

| Column | Type | Description |
|--------|------|-------------|
| `pk` | | Primary key |
| `pid` | | Person identifier |
| `role` | | Role |
| `s_ColLineage` | | Column lineage |
| `s_Generation` | | Generation tracking |
| `s_GUID` | | Globally unique identifier |
| `s_Lineage` | | Lineage tracking |

#### `TblAgentStatus`
Agent status information.

| Column | Type | Description |
|--------|------|-------------|
| `pk` | | Primary key |
| `pid` | | Person identifier |
| `AgentStatus` | | Agent status |
| `s_ColLineage` | | Column lineage |
| `s_Generation` | | Generation tracking |
| `s_GUID` | | Globally unique identifier |
| `s_Lineage` | | Lineage tracking |

#### `TblInstitutionType`
Institution type associations.

| Column | Type | Description |
|--------|------|-------------|
| `institutionid` | | Institution identifier |
| `institutiontype` | | Institution type |
| `s_ColLineage` | | Column lineage |
| `s_Generation` | | Generation tracking |
| `s_GUID` | | Globally unique identifier |
| `s_Lineage` | | Lineage tracking |

### Pleading and Outcome Tables

#### `TblPleadings`
Pleading information.

| Column | Type | Description |
|--------|------|-------------|
| `CaseID` | | Case identifier |
| `Pleading` | | Pleading text |
| `EventID` | | Event identifier |
| `ResponseNotes` | | Response notes |
| `Newkey` | | New key |
| `NewPleadingsNo` | | New pleadings number |
| `Gen_Pleading` | | Generated pleading |
| `Gen_ResponseNotes` | | Generated response notes |
| `s_ColLineage` | | Column lineage |
| `s_Generation` | | Generation tracking |
| `s_GUID` | | Globally unique identifier |
| `s_Lineage` | | Lineage tracking |

#### `TblPosteaOutcome`
Postea outcome information.

| Column | Type | Description |
|--------|------|-------------|
| `CaseID` | | Case identifier |
| `PostiaDate` | | Postea date |
| `PostiaText` | | Postea text |
| `PostiaNotes` | | Postea notes |
| `NewPosteaKey` | | New postea key |
| `NewPosteaID` | | New postea identifier |
| `AutoID` | | Auto identifier |
| `Gen_PostiaNotes` | | Generated postea notes |
| `Gen_PostiaText` | | Generated postea text |
| `s_ColLineage` | | Column lineage |
| `s_Generation` | | Generation tracking |
| `s_GUID` | | Globally unique identifier |
| `s_Lineage` | | Lineage tracking |

### Other Tables

#### `TblGoods`
Goods information for cases.

| Column | Type | Description |
|--------|------|-------------|
| `CaseID` | | Case identifier |
| `GoodsType` | | Goods type |
| `GoodsNote` | | Goods notes |
| `Gen_GoodsNote` | | Generated goods notes |
| `s_ColLineage` | | Column lineage |
| `s_Generation` | | Generation tracking |
| `s_GUID` | | Globally unique identifier |
| `s_Lineage` | | Lineage tracking |

#### `TblWritType(sub-type)`
Writ type sub-type associations.

| Column | Type | Description |
|--------|------|-------------|
| `CaseID` | | Case identifier |
| `Writ type (sub-type)` | | Writ sub-type |
| `s_ColLineage` | | Column lineage |
| `s_Generation` | | Generation tracking |
| `s_GUID` | | Globally unique identifier |
| `s_Lineage` | | Lineage tracking |

#### `London Sheriffs Court 1320`
London Sheriff's Court 1320 data.

| Column | Type | Description |
|--------|------|-------------|
| `Entry ID` | | Entry identifier |
| `Archive` | | Archive reference |
| `Document reference no` | | Document reference number |
| `Membrane no` | | Membrane number |
| `Case ID` | | Case identifier |
| `Date` | | Date |
| `Plaintiff surname` | | Plaintiff surname |
| `Defendant surname` | | Defendant surname |
| `Writ type` | | Writ type |
| `Entry type` | | Entry type |
| `Involves woman` | | Involves woman flag |
| `Notes` | | Notes |
| `Gen_TempField*0` | | Generated temporary field |
| `s_ColLineage` | | Column lineage |
| `s_Generation` | | Generation tracking |
| `s_Lineage` | | Lineage tracking |

#### `Count of Process - Pleadings`
Count of process and pleadings.

| Column | Type | Description |
|--------|------|-------------|
| `ID` | | Identifier |
| `Count` | | Count |
| `County` | | County |
| `Mesne Process or Pleaded Case` | | Mesne process or pleaded case |
| `Reference` | | Reference |
| `rot` | | Rotulet |

#### `Arbitrator Verification`
Arbitrator verification data.

| Column | Type | Description |
|--------|------|-------------|
| `id number` | | Identifier number |
| `CaseID` | | Case identifier |
| `Role` | | Role |
| `Surname` | | Surname |
| `Christian name` | | Christian name |
| `AgentOccupation` | | Agent occupation |
| `Staus` | | Status |
| `County` | | County |
| `Location` | | Location |
| `occupation supplied` | | Occupation supplied flag |
| `Status supplied` | | Status supplied flag |
| `location supplied` | | Location supplied flag |

#### `Arbitration Dummy`
Arbitration dummy table.

| Column | Type | Description |
|--------|------|-------------|
| `key` | | Key |
| `arb dummey` | | Arbitration dummy |
| `Pid` | | Person identifier |

#### `Paste Errors`
Paste errors table.

| Column | Type | Description |
|--------|------|-------------|
| `CaseID` | | Case identifier |

### Conflict Resolution Tables

Many tables have corresponding `_Conflict` tables for conflict resolution in replication scenarios. These include:
- `TblAgent_Conflict`
- `TblAgentRole_Conflict`
- `TblAgentStatus_Conflict`
- `TblCase_Conflict`
- `TblCountyandLocation_Conflict`
- `TblDueDates_Conflict`
- `TblEvent_Conflict`
- `TblEventType_Conflict`
- `TblInitialDates_Conflict`
- `TblName_Conflict`
- `TblOccupation_Conflict`
- `TblPleadings_Conflict`
- `TblPosteaOutcome_Conflict`
- `TblRelats_Conflict`

These conflict tables typically include all columns from their base tables plus:
- `ConflictRowGuid` - Conflict row identifier

---

## Notes

### Common Patterns

Both databases use different approaches:

1. **cp40_records.db**: 
   - Focused on raw scraped data with normalized entities
   - Uses many-to-many relationships for persons, places, and surnames
   - Tracks scraping progress for resumability
   - Includes Latin form processing for forenames and places

2. **cp40_database_new.sqlite**:
   - More structured database for processed case data
   - Extensive use of lookup tables for normalization
   - Includes conflict resolution tables (likely for replication)
   - Tracks detailed case information including events, pleadings, agents, and outcomes
   - Uses lineage tracking fields (`s_ColLineage`, `s_Generation`, `s_GUID`, `s_Lineage`) for data versioning

### Foreign Key Relationships

- **cp40_records.db**: Uses explicit foreign key constraints with CASCADE deletes
- **cp40_database_new.sqlite**: Relationships appear to be maintained through application logic rather than explicit foreign key constraints (based on the schema inspection)

### Data Types

Note that the schema inspection for `cp40_database_new.sqlite` did not return explicit data types for all columns. The actual database may have more specific types defined. This documentation reflects what was available from the schema inspection.

