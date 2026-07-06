"""Classification and visual styling of IEEE Organizational Units.

Two ways to classify an OU:

* ``classify_ou`` uses the API's ``type-description`` text (authoritative once a
  node has actually been fetched).
* ``classify_spoid`` uses the SPOID prefix/length rules, for "stub" nodes we
  know only by SPOID (a parent or child we haven't fetched yet). These rules
  mirror ``get_unit_type`` in the ieee-activity-report project.

Each unit type maps to a colour, a vis-network node shape, and a short emoji so
the graph is readable at a glance.
"""

from enum import Enum


class UnitType(Enum):
    REGION = "Region"
    COUNCIL = "Council"
    ZONE = "Zone"
    AREA = "Area"
    SECTION = "Section"
    SUBSECTION = "Sub-section"
    LOCAL_GROUP = "Local Group"
    CHAPTER = "Chapter"
    AFFINITY = "Affinity Group"
    STUDENT_BRANCH = "Student Branch"
    STUDENT_BRANCH_CHAPTER = "Student Branch Chapter"
    STUDENT_BRANCH_AFFINITY = "Student Branch Affinity"
    ACADEMIC = "Academic"
    SOCIETY = "Society / Technical Council"
    DIVISION = "Division"
    COMMITTEE = "Committee"
    BOARD = "Board"
    COMMUNITY = "Community"
    GROUPING = "Grouping"
    UNKNOWN = "Other"


# type -> (colour, vis-network shape, emoji, node size)
STYLE = {
    UnitType.REGION: ("#1f4e79", "hexagon", "🌐", 30),
    UnitType.COUNCIL: ("#2e8b8b", "hexagon", "🏛️", 26),
    UnitType.ZONE: ("#1565c0", "hexagon", "🧭", 26),
    UnitType.AREA: ("#3949ab", "dot", "🗺️", 22),
    UnitType.SECTION: ("#2e7d32", "dot", "📍", 24),
    UnitType.SUBSECTION: ("#66bb6a", "dot", "📌", 20),
    UnitType.LOCAL_GROUP: ("#00acc1", "dot", "🏘️", 18),
    UnitType.CHAPTER: ("#e8710a", "square", "⚙️", 18),
    UnitType.AFFINITY: ("#8e44ad", "triangle", "🤝", 18),
    UnitType.STUDENT_BRANCH: ("#c0392b", "star", "🎓", 18),
    UnitType.STUDENT_BRANCH_CHAPTER: ("#8d6e63", "square", "🎓", 16),
    UnitType.STUDENT_BRANCH_AFFINITY: ("#d81b60", "triangle", "🎓", 16),
    UnitType.ACADEMIC: ("#5c6bc0", "square", "🏫", 18),
    UnitType.SOCIETY: ("#34495e", "diamond", "🔬", 24),
    UnitType.DIVISION: ("#7f8c8d", "diamond", "🗂️", 24),
    UnitType.COMMITTEE: ("#546e7a", "square", "📋", 18),
    UnitType.BOARD: ("#455a64", "square", "🏢", 18),
    UnitType.COMMUNITY: ("#5e35b1", "dot", "🫂", 18),
    UnitType.GROUPING: ("#00897b", "diamond", "🧩", 18),
    UnitType.UNKNOWN: ("#9e9e9e", "dot", "❓", 16),
}

# Affinity-group SPOID prefixes (from ieee-activity-report affinity_groups).
AFFINITY_PREFIXES = ("WE", "LM", "YP", "SIGHT", "CN", "HKN")


def classify_ou(ou):
    """Classify a fetched OU using its API type-description text.

    Order matters: more specific descriptions ("Student Branch Chapter",
    "Sub Section") are tested before the substrings they contain.
    """
    desc = (ou.type_desc or "").lower()

    if "region" in desc:
        return UnitType.REGION
    if "council" in desc and "society" not in desc:
        # Technical councils (Sensors Council, etc.) present as societies in the
        # hierarchy; geographic councils use "Council" with no society context.
        return UnitType.COUNCIL
    if "zone" in desc:
        return UnitType.ZONE
    if "grouping" in desc:  # "Grouping" and "SBC Grouping"
        return UnitType.GROUPING
    if "area" in desc:
        return UnitType.AREA
    if "student branch chapter" in desc:
        return UnitType.STUDENT_BRANCH_CHAPTER
    if "student branch affinity" in desc:
        return UnitType.STUDENT_BRANCH_AFFINITY
    if "student branch" in desc:
        return UnitType.STUDENT_BRANCH
    if "sub section" in desc or "subsection" in desc or "sub-section" in desc:
        return UnitType.SUBSECTION
    if "section" in desc:
        return UnitType.SECTION
    if "affinity" in desc:
        return UnitType.AFFINITY
    if "chapter" in desc:
        return UnitType.CHAPTER
    if "society" in desc:
        return UnitType.SOCIETY
    if "division" in desc:
        return UnitType.DIVISION
    if "academic" in desc:  # universities / colleges
        return UnitType.ACADEMIC
    if "committee" in desc:
        return UnitType.COMMITTEE
    if "board" in desc:
        return UnitType.BOARD
    if "community" in desc:
        return UnitType.COMMUNITY
    if "local group" in desc:
        return UnitType.LOCAL_GROUP

    # Fall back to SPOID-based classification if the description is unhelpful.
    return classify_spoid(ou.spoid)


def classify_spoid(spoid):
    """Best-effort classification of a SPOID we know only by its code."""
    code = (spoid or "").strip()
    if not code:
        return UnitType.UNKNOWN

    # Geographic units: R + digits, length encodes the level.
    if code == "R10" or (len(code) == 2 and "R0" <= code <= "R9"):
        return UnitType.REGION
    if len(code) == 4 and "R000" <= code <= "R999":
        return UnitType.COUNCIL
    if len(code) == 6 and "R00000" <= code <= "R99999":
        return UnitType.SECTION
    if len(code) == 8 and "R0000000" <= code <= "R9999999":
        return UnitType.SUBSECTION

    if code.startswith("CH") and code[2:].isdigit():
        return UnitType.CHAPTER
    if code.startswith("SBC"):
        return UnitType.STUDENT_BRANCH_CHAPTER
    if code.startswith("SBA"):
        return UnitType.STUDENT_BRANCH_AFFINITY
    if code.startswith("STB"):
        return UnitType.STUDENT_BRANCH
    if code.startswith("ARR"):  # Area SPOIDs, e.g. ARR0602
        return UnitType.AREA
    if code.startswith("LGR"):  # Local Group SPOIDs, e.g. LGR60007VL
        return UnitType.LOCAL_GROUP
    # Academic units (universities): "A" + digits, or digits + dash, e.g.
    # A8636, 1-SG21PW.
    if code[:1] == "A" and code[1:2].isdigit():
        return UnitType.ACADEMIC
    if code[:1].isdigit() and "-" in code and code.split("-")[0].isdigit():
        return UnitType.ACADEMIC
    if code.startswith(AFFINITY_PREFIXES):
        return UnitType.AFFINITY

    return UnitType.UNKNOWN


def style_for(unit_type):
    """Return (colour, shape, emoji, size) for a UnitType."""
    return STYLE.get(unit_type, STYLE[UnitType.UNKNOWN])
