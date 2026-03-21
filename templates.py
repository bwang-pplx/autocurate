"""
Pre-built, tested filter templates. Qwen picks a template and provides
parameters (strings, numbers). No code generation needed.

Each template is a function that takes text + params and returns cleaned text
or a keep/drop decision.
"""


def remove_lines_containing(text, strings):
    """Remove lines that contain any of the given strings (case-insensitive)."""
    strings_lower = [s.lower() for s in strings]
    lines = text.split("\n")
    lines = [line for line in lines if not any(s in line.lower() for s in strings_lower)]
    return "\n".join(lines)


def drop_if_contains(text, strings, min_count=1):
    """Drop document if it contains at least min_count of the given strings."""
    text_lower = text.lower()
    count = sum(1 for s in strings if s.lower() in text_lower)
    return count < min_count  # True = keep


def remove_prefix_lines(text, patterns):
    """Strip leading lines that contain any of the given strings."""
    patterns_lower = [p.lower() for p in patterns]
    lines = text.split("\n")
    while lines and any(p in lines[0].lower() for p in patterns_lower):
        lines.pop(0)
    return "\n".join(lines)


def remove_suffix_lines(text, patterns):
    """Strip trailing lines that contain any of the given strings."""
    patterns_lower = [p.lower() for p in patterns]
    lines = text.split("\n")
    while lines and any(p in lines[-1].lower() for p in patterns_lower):
        lines.pop()
    return "\n".join(lines)


def drop_short_docs(text, min_chars=200):
    """Drop documents shorter than min_chars."""
    return len(text.strip()) >= min_chars  # True = keep


def drop_if_keyword_density(text, keywords, max_density=0.05):
    """Drop document if keyword density (keyword occurrences / total words) exceeds threshold."""
    words = text.lower().split()
    if not words:
        return False
    keywords_lower = [k.lower() for k in keywords]
    hits = sum(1 for w in words if any(k in w for k in keywords_lower))
    density = hits / len(words)
    return density <= max_density  # True = keep


def replace_strings(text, replacements):
    """Replace exact strings. replacements is list of [find, replace] pairs."""
    for find, replace in replacements:
        text = text.replace(find, replace)
    return text


def remove_duplicate_lines(text):
    """Remove exact duplicate non-empty lines, keeping first occurrence."""
    seen = set()
    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            lines.append(line)
            continue
        if stripped in seen:
            continue
        seen.add(stripped)
        lines.append(line)
    return "\n".join(lines)


def drop_by_language_markers(text, foreign_strings, max_density=0.1):
    """Drop docs with too many foreign language markers."""
    words = text.lower().split()
    if not words:
        return False
    foreign_lower = [s.lower() for s in foreign_strings]
    hits = sum(1 for w in words if w in foreign_lower)
    return hits / len(words) <= max_density  # True = keep


# Registry of all templates
TEMPLATES = {
    "REMOVE_LINES_CONTAINING": {
        "fn": remove_lines_containing,
        "type": "cleaner",
        "params": "strings: list of strings to match",
        "description": "Remove lines containing any of the given strings",
    },
    "DROP_IF_CONTAINS": {
        "fn": drop_if_contains,
        "type": "filter",
        "params": "strings: list of strings; min_count: how many must match to drop (default 1)",
        "description": "Drop whole document if it contains N+ of these strings",
    },
    "REMOVE_PREFIX_LINES": {
        "fn": remove_prefix_lines,
        "type": "cleaner",
        "params": "patterns: list of strings to match in leading lines",
        "description": "Strip leading lines matching patterns (navbars, breadcrumbs)",
    },
    "REMOVE_SUFFIX_LINES": {
        "fn": remove_suffix_lines,
        "type": "cleaner",
        "params": "patterns: list of strings to match in trailing lines",
        "description": "Strip trailing lines matching patterns (footers, disclaimers)",
    },
    "DROP_SHORT_DOCS": {
        "fn": drop_short_docs,
        "type": "filter",
        "params": "min_chars: minimum character count (default 200)",
        "description": "Drop documents shorter than min_chars",
    },
    "DROP_IF_KEYWORD_DENSITY": {
        "fn": drop_if_keyword_density,
        "type": "filter",
        "params": "keywords: list of spam/junk keywords; max_density: threshold (default 0.05)",
        "description": "Drop document if keyword density exceeds threshold",
    },
    "REPLACE_STRINGS": {
        "fn": replace_strings,
        "type": "cleaner",
        "params": "replacements: list of [find, replace] pairs",
        "description": "Replace exact strings",
    },
    "REMOVE_DUPLICATE_LINES": {
        "fn": remove_duplicate_lines,
        "type": "cleaner",
        "params": "(none)",
        "description": "Remove exact duplicate lines",
    },
    "DROP_BY_LANGUAGE_MARKERS": {
        "fn": drop_by_language_markers,
        "type": "filter",
        "params": "foreign_strings: list of common foreign words; max_density: threshold (default 0.1)",
        "description": "Drop docs with too many foreign language words",
    },
}


def get_template_menu():
    """Format templates as a string for the Qwen prompt."""
    lines = ["Available templates:"]
    for name, info in TEMPLATES.items():
        lines.append(f"  {name} ({info['type']}): {info['description']}")
        lines.append(f"    Params: {info['params']}")
    return "\n".join(lines)
