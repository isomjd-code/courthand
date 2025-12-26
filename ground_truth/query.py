"""
SQL queries used for extracting ground truth records.

This module contains the complex SQL query used to extract comprehensive
case data from the CP40 database as structured JSON.
"""

# Complex SQL query that extracts comprehensive case data as a JSON object.
#
# This query constructs a nested JSON structure containing:
# - TblCase: Basic case information (rotulus, county, damages, writ type, term, notes)
# - TblCaseType: Array of case types associated with the case
# - Agents: Array of all parties/agents with:
#   - Name information (Christian name, surname, suffix)
#   - Role (plaintiff, defendant, attorney, etc.)
#   - Gender and institution name
#   - Occupation
#   - Status (alive, deceased, etc.)
#   - Location details (place, county, parish, ward, country)
# - TblEvents: Array of events with:
#   - Event types
#   - Value details (amount, description, shillings)
#   - Location information
#   - Initial dates
# - TblPleadings: Array of pleadings with text and response notes
# - TblPostea: Array of postea outcomes with text, dates, and notes
#
# The query takes a single parameter: CaseID (bound via ? placeholder).
#
# Returns a single row with a JSON string that can be parsed into a Python dict.
JSON_QUERY = """
SELECT json_object(
    'TblCase', json_object(
        'CaseRot', c.CaseRot,
        'County', c.County,
        'DamClaimed', c.DamClaimed,
        'DamAward', c.DamAward,
        'WritType', (
            SELECT "Writ type (sub-type)"
            FROM "TblWritType(sub-type)"
            WHERE CaseID = c.CaseID
            LIMIT 1
        ),
        'CaseNotes', c.CaseNotes,
        'Term', (
            SELECT r.dateterm || ' ' || r.dateyear
            FROM TblReference r
            WHERE r.docid = c.DocID
        )
    ),
    'TblCaseType', json_object(
        'CaseType', (
            SELECT json_group_array(ct.CaseType)
            FROM TblCaseType ct
            WHERE ct.CaseID = c.CaseID
        )
    ),
    'Agents', (
        SELECT json_group_array(
            json_object(
                'TblName', json_object(
                    'Christian_name', n."Christian name",
                    'Surname', n.Surname,
                    'Suffix', n.Suffix
                ),
                'TblAgentRole', json_object(
                    'role', (
                        SELECT lar.DisplayRole
                        FROM TblAgentRole ar
                        JOIN LookupAgentRole lar ON ar.role = lar.role
                        WHERE ar.pid = n.pid
                        LIMIT 1
                    )
                ),
                'TblAgent', json_object(
                    'AgentGender', (
                        SELECT a.AgentGender
                        FROM TblAgent a
                        WHERE a.pid = n.pid AND a.CaseID = c.CaseID
                        LIMIT 1
                    ),
                    'InstitutionName', (
                        SELECT a.InstitutionName
                        FROM TblAgent a
                        WHERE a.pid = n.pid AND a.CaseID = c.CaseID
                        LIMIT 1
                    ),
                    'Occupation', (
                        SELECT o.AgentOccupation
                        FROM TblOccupation o
                        WHERE o.pid = n.pid
                        LIMIT 1
                    ),
                    'AgentStatus', (
                        SELECT CASE
                            WHEN (
                                SELECT "Agent Deceased"
                                FROM TblAgent
                                WHERE pid = n.pid AND CaseID = c.CaseID
                            ) = 1 THEN 'dec.'
                            ELSE (
                                SELECT ast.AgentStatus
                                FROM TblAgentStatus ast
                                WHERE ast.pid = n.pid
                                LIMIT 1
                            )
                        END
                    ),
                    'LocationDetails', json_object(
                        'SpecificPlace', (
                            SELECT CASE
                                WHEN loc.Nuper = 1 THEN '(lately of) ' || loc.Location
                                ELSE loc.Location
                            END
                            FROM TblCountyandLocation loc
                            WHERE loc.pid = n.pid
                            LIMIT 1
                        ),
                        'County', (
                            SELECT loc.County
                            FROM TblCountyandLocation loc
                            WHERE loc.pid = n.pid
                            LIMIT 1
                        ),
                        'Parish', (
                            SELECT a.AgentParish
                            FROM TblAgent a
                            WHERE a.pid = n.pid AND a.CaseID = c.CaseID
                            LIMIT 1
                        ),
                        'Ward', (
                            SELECT a.AgentWard
                            FROM TblAgent a
                            WHERE a.pid = n.pid AND a.CaseID = c.CaseID
                            LIMIT 1
                        ),
                        'Country', (
                            SELECT a.AgentCountry
                            FROM TblAgent a
                            WHERE a.pid = n.pid AND a.CaseID = c.CaseID
                            LIMIT 1
                        )
                    )
                )
            )
        )
        FROM TblName n
        WHERE n.NameCaseID = c.CaseID
    ),
    'TblEvents', (
        SELECT json_group_array(
            json_object(
                'EventType', (
                    SELECT json_group_array(et.EventType)
                    FROM TblEventType et
                    WHERE et.EventID = e.EventID
                ),
                'EventDetails', json_object(
                    'ValueAmount', e.value,
                    'ValueDescription', e.EventNotes,
                    'ValueShillings', e."value in approximate shillings"
                ),
                'LocationDetails', json_object(
                    'SpecificPlace', e.EventPlaceMs,
                    'Parish', e.EventPlaceParish,
                    'Ward', e.EventPlaceWard,
                    'County', e.EventPlaceCounty,
                    'Country', e.EventPlaceCountry
                ),
                'EventDate', (
                    SELECT json_group_array(
                        json_object(
                            'Date', CAST(id.InitialDate AS TEXT),
                            'DateType', 'initial'
                        )
                    )
                    FROM TblInitialDates id
                    WHERE id.EventID = e.EventID
                )
            )
        )
        FROM TblEvent e
        WHERE e.CaseID = c.CaseID
    ),
    'TblPleadings', (
        SELECT json_group_array(
            json_object(
                'PleadingText', p.Pleading,
                'ResponseNotes', p.ResponseNotes
            )
        )
        FROM TblPleadings p
        WHERE p.CaseID = c.CaseID
    ),
    'TblPostea', (
        SELECT json_group_array(
            json_object(
                'PosteaText', po.PostiaText,
                'Date', po.PostiaDate,
                'Notes', po.PostiaNotes
            )
        )
        FROM TblPosteaOutcome po
        WHERE po.CaseID = c.CaseID
    )
)
FROM TblCase c
WHERE c.CaseID = ?;
"""

