from typing import List, Optional, Set, Tuple


__all__ = ["generate_table"]


def generate_table(
        header: List[str],
        data: List[List[str]],
        alignments: Optional[List[str]] = None,
        horizontal_lines: Optional[List[bool]] = None,
        bold_cells: Optional[Set[Tuple[int, int]]] = None,
        fmt: str = "markdown"
) -> str:
    assert fmt in {"markdown", "latex"}

    assert all(len(header) == len(item) for item in data)

    if alignments is None:
        alignments = ["left"] + ["right"] * (len(header) - 1)

    if horizontal_lines is None or fmt == "markdown":
        horizontal_lines = [False] * len(data)
    horizontal_lines[-1] = fmt == "latex"  # always a horizontal line after last line for latex, but not for markdown

    if bold_cells is None:
        bold_cells = [[False] * len(item) for item in data]
    else:
        bold_cells = [[(i, j) in bold_cells for j in range(len(data[i]))] for i in range(len(data))]

    tables_lines = []

    opening_str = _open_table(fmt, alignments)
    if opening_str:
        tables_lines.append(opening_str)

    header_str = _table_row(fmt, header, [False] * len(header)) + _table_horizontal_line(fmt, len(header), alignments)
    tables_lines.append(header_str)

    for item, horizontal_line, bold in zip(data, horizontal_lines, bold_cells):
        line = _table_row(fmt, item, bold)
        if horizontal_line:
            line += _table_horizontal_line(fmt, len(item), alignments)
        tables_lines.append(line)

    closing_str = _close_table(fmt)
    if closing_str:
        tables_lines.append(closing_str)

    return "\n".join(tables_lines)


_MARK_DOWN_ALIGNMENTS = {
    "center": ":---:",
    "left": ":--",
    "right": "--:"
}

_LATEX_ALIGNMENTS = {
    "center": "c",
    "left": "l",
    "right": "r"
}


def _open_table(fmt: str, alignments: List[str]) -> str:
    if fmt == "markdown":
        return ""
    else:
        return "\\begin{tabular}{" + "".join(_LATEX_ALIGNMENTS[align] for align in alignments) + "} \\hline"


def _close_table(fmt: str) -> str:
    if fmt == "markdown":
        return ""
    else:
        return "\\end{tabular}"


_LATEX_ESCAPE_CHARS = {"&", "%", "$", "#", "_", "{", "}"}


def _format_latex(s: str, bold: bool) -> str:
    s = "".join("\\" + char if char in _LATEX_ESCAPE_CHARS else char for char in s)
    if bold:
        s = "\\textbf{" + s + "}"
    return s


def _format_markdown(s: str, bold: bool) -> str:
    if bold:
        s = "**" + s + "**"
    return s


def _table_row(fmt: str, data: List[str], bold: List[bool]) -> str:
    assert len(data) == len(bold)

    if fmt == "markdown":
        return "| " + " | ".join(_format_markdown(s, b) for s, b in zip(data, bold)) + " |"
    else:
        return " & ".join(_format_latex(s, b) for s, b in zip(data, bold)) + " \\\\ "


def _table_horizontal_line(fmt: str, num_cols: int, alignments: List[str]) -> str:
    assert num_cols == len(alignments) and all(align in {"left", "right", "center"} for align in alignments)

    if fmt == "markdown":
        return "\n| " + " | ".join(_MARK_DOWN_ALIGNMENTS[align] for align in alignments) + " |"
    else:
        return "\\hline"
