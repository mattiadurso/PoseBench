"""Characterization test pinning print_table's exact rendered output.

Golden lines (incl. trailing spaces) lock the formatting -- numeric auto-detect,
float formatting, fill_missing, min/clip widths, and the width-is-computed-before-
clipping behavior -- so the helper-extraction refactor is provably equivalent.
"""

from benchmarks_2D.utils_benchmark import print_table

ROWS = [
    {"Method": "disk", "auc": 0.12345, "n": 3},
    {"Method": "a-very-long-method-name", "auc": 0.5},  # missing "n" -> fill_missing
    {"Method": "x", "auc": 12.0, "n": 12000},
]

GOLDEN_A = [
    "Method                  |   auc |     n",
    "----------------------- | ----- | -----",
    "disk                    |  0.12 |     3",
    "a-very-long…            |  0.50 |     -",
    "x                       | 12.00 | 12000",
]

GOLDEN_B = [
    "                   disk |     3",
    "a-very-long-method-name |      ",
    "                      x | 12000",
]


def test_print_table_default_formatting(capsys):
    print_table(
        ROWS,
        min_widths={"Method": 8},
        clip_widths={"Method": 12},
        float_format=".2f",
        fill_missing="-",
    )
    assert capsys.readouterr().out.splitlines() == GOLDEN_A


def test_print_table_forced_align_no_header(capsys):
    print_table(ROWS, columns=["Method", "n"], right_align={"Method"}, header=False)
    assert capsys.readouterr().out.splitlines() == GOLDEN_B


def test_print_table_empty(capsys):
    print_table([])
    assert capsys.readouterr().out.strip() == "(empty)"
