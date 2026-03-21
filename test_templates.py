"""Tests for filter templates."""

from templates import *


def test_remove_lines_containing():
    text = "Good line\nAccepter cookies her\nAnother good line\nKlik her for mere"
    result = remove_lines_containing(text, ["Accepter cookies", "Klik her"])
    assert result == "Good line\nAnother good line"

    # Case insensitive
    result = remove_lines_containing(text, ["accepter COOKIES"])
    assert result == "Good line\nAnother good line\nKlik her for mere"

    # No match
    result = remove_lines_containing(text, ["xyz"])
    assert result == text

    # Empty text
    assert remove_lines_containing("", ["foo"]) == ""


def test_drop_if_contains():
    text = "Buy now! Great price! Free shipping!"
    assert drop_if_contains(text, ["buy now", "free shipping"], min_count=2) == False
    assert drop_if_contains(text, ["buy now", "free shipping"], min_count=3) == True
    assert drop_if_contains(text, ["xyz"], min_count=1) == True

    # Default min_count=1
    assert drop_if_contains(text, ["buy now"]) == False
    assert drop_if_contains(text, ["xyz"]) == True


def test_remove_prefix_lines():
    text = "Forside > Produkter > Sko\nBreadcrumb trail\nActual content here"
    result = remove_prefix_lines(text, ["Forside", "Breadcrumb"])
    assert result == "Actual content here"

    # No match
    result = remove_prefix_lines(text, ["xyz"])
    assert result == text

    # Empty
    assert remove_prefix_lines("", ["foo"]) == ""


def test_remove_suffix_lines():
    text = "Actual content\nGood stuff\nCopyright 2024\nKontakt os"
    result = remove_suffix_lines(text, ["Copyright", "Kontakt"])
    assert result == "Actual content\nGood stuff"

    # No match at end
    result = remove_suffix_lines(text, ["xyz"])
    assert result == text


def test_drop_short_docs():
    assert drop_short_docs("short", min_chars=200) == False
    assert drop_short_docs("a" * 200, min_chars=200) == True
    assert drop_short_docs("a" * 199, min_chars=200) == False
    assert drop_short_docs("  \n  ", min_chars=1) == False  # whitespace only


def test_drop_if_keyword_density():
    text = "køb billigt køb nu gratis tilbud køb her"  # 8 words, 3 "køb"
    assert drop_if_keyword_density(text, ["køb"], max_density=0.3) == False
    assert drop_if_keyword_density(text, ["køb"], max_density=0.5) == True

    # Normal text
    normal = "Danmark er et land i Nordeuropa med mange interessante steder"
    assert drop_if_keyword_density(normal, ["køb", "gratis"], max_density=0.05) == True

    # Empty
    assert drop_if_keyword_density("", ["foo"], max_density=0.05) == False


def test_replace_strings():
    text = "Læs mere her. Klik her for info."
    result = replace_strings(text, [["Læs mere her.", ""], ["Klik her for info.", ""]])
    assert result == " "

    # No match
    result = replace_strings(text, [["xyz", "abc"]])
    assert result == text


def test_remove_duplicate_lines():
    text = "Line A\nLine B\nLine A\nLine C\nLine B\nLine D"
    result = remove_duplicate_lines(text)
    assert result == "Line A\nLine B\nLine C\nLine D"

    # Empty lines preserved
    text = "A\n\nB\n\nC"
    result = remove_duplicate_lines(text)
    assert result == "A\n\nB\n\nC"


def test_drop_by_language_markers():
    # Mostly Danish
    danish = "Dette er en dansk tekst om mange forskellige emner i Danmark"
    assert drop_by_language_markers(danish, ["the", "and", "is", "of"], max_density=0.1) == True

    # Mostly English
    english = "This is the best and the most important of the things"
    assert drop_by_language_markers(english, ["the", "and", "is", "of"], max_density=0.1) == False

    # Empty
    assert drop_by_language_markers("", ["the"], max_density=0.1) == False


if __name__ == "__main__":
    tests = [f for f in dir() if f.startswith("test_")]
    passed = 0
    failed = 0
    for t in tests:
        try:
            globals()[t]()
            print(f"  PASS {t}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL {t}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {t}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
