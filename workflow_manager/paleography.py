"""Paleography utilities."""

from __future__ import annotations

from typing import Dict, List, Tuple


class PaleographyMatcher:
    """
    Weighted Levenshtein distance calculator optimized for medieval Latin Court Hand.

    Implements fuzzy string matching with paleographic awareness, accounting for:
    - Common character confusions in medieval handwriting (i/j, u/v, r/ꝛ, etc.)
    - Abbreviation marks and special characters
    - Case variations and ligatures
    - Tiered substitution costs based on visual similarity

    The matcher uses weighted edit distances where substitutions between visually
    similar characters cost less than substitutions between dissimilar ones.
    """

    def __init__(self) -> None:
        """
        Initialize the PaleographyMatcher.

        Builds internal character maps and special pair mappings for weighted
        distance calculations. This initialization is called once and the
        matcher can be reused for multiple comparisons.
        """
        self.char_map: Dict[str, Tuple[int, frozenset[str]]] = {}
        self.special_pairs: Dict[frozenset[str], float] = {}
        self._build_lowercase_sets()
        self._build_uppercase_sets()
        self._build_abbreviation_sets()
        self._build_special_pairs()

    def _build_lowercase_sets(self) -> None:
        """
        Build character equivalence sets for lowercase letters.

        Creates tiered groups of visually similar lowercase characters:
        - Tier 1: Very similar (low substitution cost)
        - Tier 2: Moderately similar (medium substitution cost)
        - Tier 3: Somewhat similar (higher substitution cost)
        """
        tier1_sets = [
            {"i", "j", "ı", "n", "u", "ū", "ī", "ń"},
            {"f", "ſ"},
            {"c", "t"},
            {"r", "ꝛ"},
            {"u", "v", "ꝟ"},
            {"i", "j"},
        ]
        tier2_sets = [
            {"m", "w", "nn", "iu", "in", "ni", "nu", "un"},
            {"a", "o", "e", "ö", "ë", "ä"},
            {"a", "u", "ci"},
            {"b", "l", "h"},
            {"d", "đ", "cl", "ꝺ"},
            {"g", "q", "y"},
            {"y", "ȝ", "ʒ", "z"},
            {"e", "o"},
            {"w", "ƿ", "p"},
        ]
        tier3_sets = [
            {"k", "h", "b", "lt"},
            {"s", "ſ", "ʃ"},
            {"x", "y"},
            {"þ", "ð", "y", "p"},
            {"ð", "d", "o"},
        ]

        for v_set in tier1_sets:
            for char in v_set:
                self.char_map[char] = (1, frozenset(v_set))

        for v_set in tier2_sets:
            for char in v_set:
                if char not in self.char_map or self.char_map[char][0] > 1:
                    self.char_map[char] = (2, frozenset(v_set))

        for v_set in tier3_sets:
            for char in v_set:
                if char not in self.char_map or self.char_map[char][0] > 2:
                    self.char_map[char] = (3, frozenset(v_set))

    def _build_uppercase_sets(self) -> None:
        """
        Build character equivalence sets for uppercase letters.

        Creates tiered groups of visually similar uppercase characters,
        following the same tier system as lowercase sets.
        """
        uppercase_sets = [
            (1, {"B", "G", "W"}),
            (1, {"I", "J"}),
            (1, {"U", "V", "Ꝟ"}),
            (2, {"C", "E", "G", "O", "T"}),
            (2, {"B", "D", "P", "R", "S"}),
            (2, {"L", "S", "J", "I"}),
            (2, {"Þ", "P", "Y"}),
            (2, {"Ð", "D", "O"}),
            (2, {"Ȝ", "Z", "Y", "G"}),
            (2, {"Ƿ", "P", "W"}),
            (3, {"A", "H", "K", "M", "N"}),
            (3, {"O", "Q", "G"}),
            (3, {"F", "I", "J", "L", "T"}),
            (3, {"X", "Y", "Z"}),
        ]

        for tier, v_set in uppercase_sets:
            for char in v_set:
                if char not in self.char_map:
                    self.char_map[char] = (tier, frozenset(v_set))
                elif self.char_map[char][0] > tier:
                    self.char_map[char] = (tier, frozenset(v_set))

    def _build_abbreviation_sets(self) -> None:
        """
        Build character equivalence sets for medieval abbreviation marks.

        Groups special characters used in medieval abbreviations (e.g., ꝑ, ꝓ, ꝫ)
        with their base letter forms for matching purposes.
        """
        abbreviation_sets = [
            (1, {"p", "ꝑ", "ꝓ", "ꝕ"}),
            (1, {"P", "Ꝑ", "Ꝓ", "Ꝕ"}),
            (1, {"q", "ꝗ", "ꝙ"}),
            (1, {"Q", "Ꝗ", "Ꝙ"}),
            (1, {"r", "ꝛ"}),
            (1, {"R", "Ꝛ"}),
            (2, {"ꝝ", "r", "ꝛ"}),
            (2, {"Ꝝ", "R", "Ꝛ"}),
            (1, {"ꝫ", "ꝰ", "ꝯ", "⁊"}),
            (1, {"Ꝫ", "Ꝯ"}),
            (1, {"v", "ꝟ", "u"}),
            (1, {"V", "Ꝟ", "U"}),
        ]

        for tier, v_set in abbreviation_sets:
            for char in v_set:
                if char not in self.char_map:
                    self.char_map[char] = (tier, frozenset(v_set))

    def _build_special_pairs(self) -> None:
        """
        Build special pair mappings with custom substitution costs.

        Defines specific character pairs with fine-tuned substitution costs
        based on paleographic analysis of common confusions in Court Hand.
        """
        self.special_pairs = {
            frozenset({"B", "G"}): 0.2,
            frozenset({"B", "W"}): 0.2,
            frozenset({"G", "W"}): 0.2,
            frozenset({"B", "R"}): 0.3,
            frozenset({"G", "C"}): 0.3,
            frozenset({"G", "S"}): 0.35,
            frozenset({"D", "O"}): 0.35,
            frozenset({"D", "B"}): 0.35,
            frozenset({"E", "C"}): 0.25,
            frozenset({"T", "C"}): 0.25,
            frozenset({"T", "I"}): 0.35,
            frozenset({"P", "R"}): 0.35,
            frozenset({"S", "L"}): 0.4,
            frozenset({"F", "T"}): 0.35,
            frozenset({"H", "K"}): 0.35,
            frozenset({"M", "N"}): 0.3,
            frozenset({"N", "U"}): 0.35,
            frozenset({"I", "J"}): 0.1,
            frozenset({"U", "V"}): 0.1,
            frozenset({"U", "Ꝟ"}): 0.15,
            frozenset({"V", "Ꝟ"}): 0.1,
            frozenset({"Þ", "P"}): 0.3,
            frozenset({"Þ", "Y"}): 0.2,
            frozenset({"Ð", "D"}): 0.25,
            frozenset({"Ð", "O"}): 0.4,
            frozenset({"Ȝ", "Z"}): 0.2,
            frozenset({"Ȝ", "Y"}): 0.3,
            frozenset({"Ȝ", "G"}): 0.35,
            frozenset({"Ƿ", "P"}): 0.25,
            frozenset({"Ƿ", "W"}): 0.3,
            frozenset({"P", "Ꝑ"}): 0.15,
            frozenset({"P", "Ꝓ"}): 0.2,
            frozenset({"P", "Ꝕ"}): 0.2,
            frozenset({"Ꝑ", "Ꝓ"}): 0.25,
            frozenset({"Q", "Ꝗ"}): 0.15,
            frozenset({"Q", "Ꝙ"}): 0.2,
            frozenset({"Ꝗ", "Ꝙ"}): 0.25,
            frozenset({"R", "Ꝛ"}): 0.15,
            frozenset({"R", "Ꝝ"}): 0.25,
            frozenset({"Ꝛ", "Ꝝ"}): 0.3,
            frozenset({"Ꝫ", "Ꝯ"}): 0.3,
            frozenset({"f", "ſ"}): 0.15,
            frozenset({"c", "t"}): 0.2,
            frozenset({"u", "v"}): 0.1,
            frozenset({"i", "j"}): 0.1,
            frozenset({"u", "ꝟ"}): 0.15,
            frozenset({"v", "ꝟ"}): 0.1,
            frozenset({"n", "u"}): 0.15,
            frozenset({"n", "v"}): 0.2,
            frozenset({"m", "in"}): 0.2,
            frozenset({"m", "ni"}): 0.2,
            frozenset({"m", "iu"}): 0.2,
            frozenset({"nn", "m"}): 0.2,
            frozenset({"w", "uu"}): 0.2,
            frozenset({"w", "vv"}): 0.2,
            frozenset({"b", "l"}): 0.3,
            frozenset({"b", "h"}): 0.35,
            frozenset({"h", "k"}): 0.35,
            frozenset({"l", "i"}): 0.4,
            frozenset({"g", "q"}): 0.3,
            frozenset({"g", "y"}): 0.35,
            frozenset({"p", "q"}): 0.4,
            frozenset({"a", "o"}): 0.3,
            frozenset({"a", "u"}): 0.35,
            frozenset({"e", "o"}): 0.35,
            frozenset({"e", "c"}): 0.3,
            frozenset({"d", "cl"}): 0.25,
            frozenset({"d", "e"}): 0.4,
            frozenset({"d", "a"}): 0.4,
            frozenset({"þ", "y"}): 0.2,
            frozenset({"þ", "p"}): 0.35,
            frozenset({"ð", "d"}): 0.3,
            frozenset({"ð", "o"}): 0.4,
            frozenset({"ȝ", "z"}): 0.25,
            frozenset({"ȝ", "y"}): 0.3,
            frozenset({"ȝ", "g"}): 0.35,
            frozenset({"ƿ", "p"}): 0.25,
            frozenset({"ƿ", "w"}): 0.3,
            frozenset({"ſ", "l"}): 0.4,
            frozenset({"ſ", "s"}): 0.3,
            frozenset({"ſſ", "ff"}): 0.25,
            frozenset({"p", "ꝑ"}): 0.15,
            frozenset({"p", "ꝓ"}): 0.2,
            frozenset({"p", "ꝕ"}): 0.2,
            frozenset({"ꝑ", "ꝓ"}): 0.25,
            frozenset({"q", "ꝗ"}): 0.15,
            frozenset({"q", "ꝙ"}): 0.2,
            frozenset({"ꝗ", "ꝙ"}): 0.25,
            frozenset({"r", "ꝛ"}): 0.15,
            frozenset({"r", "ꝝ"}): 0.25,
            frozenset({"ꝛ", "ꝝ"}): 0.3,
            frozenset({"ꝛ", "2"}): 0.3,
            frozenset({"ꝫ", "ꝰ"}): 0.2,
            frozenset({"ꝫ", "ꝯ"}): 0.3,
            frozenset({"ꝫ", "⁊"}): 0.35,
            frozenset({"ꝰ", "ꝯ"}): 0.3,
            frozenset({"ꝯ", "⁊"}): 0.35,
            frozenset({"ꝫ", "z"}): 0.5,
            frozenset({"ꝫ", "3"}): 0.4,
            frozenset({"Ꝫ", "Z"}): 0.5,
            frozenset({"c", "C"}): 0.15,
            frozenset({"o", "O"}): 0.15,
            frozenset({"s", "S"}): 0.2,
            frozenset({"w", "W"}): 0.2,
            frozenset({"x", "X"}): 0.15,
            frozenset({"z", "Z"}): 0.15,
            frozenset({"ꝫ", "Ꝫ"}): 0.1,
            frozenset({"ꝯ", "Ꝯ"}): 0.1,
            frozenset({"ff", "F"}): 0.25,
            frozenset({"ff", "H"}): 0.4,
        }

    def get_substitution_cost(self, c1: str, c2: str) -> float:
        """
        Calculate the substitution cost between two characters.

        Returns a cost between 0.0 (identical) and 1.0 (completely different)
        based on paleographic similarity. Uses character equivalence sets
        and special pair mappings.

        Args:
            c1: First character to compare.
            c2: Second character to compare.

        Returns:
            Substitution cost as a float:
            - 0.0: Identical characters
            - 0.1: Same letter, different case
            - 0.15-0.5: Characters in same equivalence set (tier-dependent)
            - 0.1-0.5: Special pair mappings (custom costs)
            - 1.0: Completely different characters
        """
        if c1 == c2:
            return 0.0

        c1_lower, c2_lower = c1.lower(), c2.lower()
        if c1_lower == c2_lower:
            return 0.1

        pair = frozenset({c1, c2})
        if pair in self.special_pairs:
            return self.special_pairs[pair]

        pair_lower = frozenset({c1_lower, c2_lower})
        if pair_lower in self.special_pairs and pair_lower != pair:
            return self.special_pairs[pair_lower] + 0.05

        entry1 = self.char_map.get(c1)
        entry2 = self.char_map.get(c2)

        if entry1 and entry2:
            tier1, set1 = entry1
            tier2, set2 = entry2

            if set1 == set2:
                if tier1 == 1:
                    return 0.2
                if tier1 == 2:
                    return 0.35
                return 0.5

            if tier1 == tier2:
                return 0.7

        return 1.0

    def weighted_levenshtein(self, s1: str, s2: str) -> float:
        """
        Calculate weighted Levenshtein distance between two strings.

        Uses paleographic-aware substitution costs and applies prefix/suffix
        matching bonuses to reduce distance for strings that share beginnings
        or endings.

        Args:
            s1: First string to compare.
            s2: Second string to compare.

        Returns:
            Weighted edit distance as a float. Lower values indicate greater similarity.
            The distance can be negative if prefix/suffix bonuses exceed the base distance.
        """
        if len(s1) < len(s2):
            return self.weighted_levenshtein(s2, s1)

        if not s2:
            return float(len(s1)) * 0.8

        prefix_match_bonus = 0.0
        min_len = min(len(s1), len(s2))

        if min_len >= 5 and s1[:5] == s2[:5]:
            prefix_match_bonus = -2.0
        elif min_len >= 4 and s1[:4] == s2[:4]:
            prefix_match_bonus = -1.5
        elif min_len >= 3 and s1[:3] == s2[:3]:
            prefix_match_bonus = -0.8
        elif min_len >= 2 and s1[:2] == s2[:2]:
            prefix_match_bonus = -0.3

        if min_len >= 3 and s1[-3:] == s2[-3:]:
            prefix_match_bonus -= 0.5

        ins_del_cost = 0.8
        prev_row = [float(x) * ins_del_cost for x in range(len(s2) + 1)]

        for i, c1 in enumerate(s1):
            curr_row = [float(i + 1) * ins_del_cost]
            for j, c2 in enumerate(s2):
                sub_cost = self.get_substitution_cost(c1, c2)
                curr_row.append(
                    min(
                        prev_row[j + 1] + ins_del_cost,
                        curr_row[j] + ins_del_cost,
                        prev_row[j] + sub_cost,
                    )
                )
            prev_row = curr_row

        return max(0.0, prev_row[-1] + prefix_match_bonus)

    def similarity_score(self, s1: str, s2: str) -> float:
        """
        Calculate similarity score between two strings (0.0 to 1.0).

        Converts weighted Levenshtein distance into a normalized similarity score
        where 1.0 indicates identical strings and 0.0 indicates completely different.

        Args:
            s1: First string to compare.
            s2: Second string to compare.

        Returns:
            Similarity score between 0.0 and 1.0:
            - 1.0: Identical strings (or both empty)
            - 0.0: One string is empty and the other is not
            - 0.0-1.0: Normalized similarity based on distance
        """
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0

        distance = self.weighted_levenshtein(s1, s2)
        max_len = max(len(s1), len(s2))
        return max(0.0, 1.0 - (distance / max_len))

    def find_best_match(
        self, query: str, candidates: List[str], threshold: float = 0.5
    ) -> List[Tuple[str, float]]:
        """
        Find all candidate strings that match a query above a similarity threshold.

        Args:
            query: The string to search for.
            candidates: List of candidate strings to search.
            threshold: Minimum similarity score (0.0-1.0) to include. Defaults to 0.5.

        Returns:
            List of (candidate, score) tuples, sorted by score (highest first).
            Only includes candidates with similarity >= threshold.
        """
        results = []
        for candidate in candidates:
            score = self.similarity_score(query, candidate)
            if score >= threshold:
                results.append((candidate, score))
        return sorted(results, key=lambda x: x[1], reverse=True)

