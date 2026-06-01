import os
import re
import argparse
from typing import List, Optional
from dataclasses import dataclass


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class Block:
    type: str            # "heading", "table", "text"
    lines: List[str]
    heading: str = ""


# -----------------------------
# Markdown table detection
# -----------------------------

def is_separator_row(line: str) -> bool:
    if "|" not in line:
        return False

    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    if not cells:
        return False

    for c in cells:
        if not c:
            return False
        if not all(ch in "-:" for ch in c):
            return False

    return True


def parse_markdown_blocks(md: str) -> List[Block]:
    lines = md.splitlines()
    blocks: List[Block] = []

    i = 0
    current_heading = ""

    while i < len(lines):
        line = lines[i]

        # Heading
        if re.match(r"^#{1,6}\s+", line):
            current_heading = line.strip()
            blocks.append(Block("heading", [line], current_heading))
            i += 1
            continue

        # Table: header + separator
        if (
            i + 1 < len(lines)
            and line.strip().startswith("|")
            and lines[i + 1].strip().startswith("|")
            and is_separator_row(lines[i + 1])
        ):
            table_lines = [lines[i], lines[i + 1]]
            i += 2

            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1

            blocks.append(Block("table", table_lines, current_heading))
            continue

        # Normal text
        blocks.append(Block("text", [line], current_heading))
        i += 1

    return blocks


# -----------------------------
# Table quality helpers
# -----------------------------

def count_columns(row: str) -> int:
    return len(row.strip("|").split("|"))


def normalize_cell(cell: str) -> str:
    return cell.strip().replace("\xa0", "").replace("\u200b", "")


def empty_cell_ratio(table_lines: List[str]) -> float:
    cells = []
    for row in table_lines:
        cells.extend(normalize_cell(c) for c in row.strip("|").split("|"))
    empty = sum(1 for c in cells if not c)
    return empty / max(len(cells), 1)


def has_repeated_cells(table_lines: List[str]) -> bool:
    seen = set()
    repeats = 0
    for row in table_lines:
        for cell in row.strip("|").split("|"):
            token = normalize_cell(cell)
            if token:
                if token in seen:
                    repeats += 1
                seen.add(token)
    return repeats > 3


# ✅ **SEMANTIC vertical merge detector (FIXED)**
def has_vertical_merge_collapse(table_lines: List[str]) -> bool:
    if len(table_lines) < 4:
        return False

    rows = [
        [normalize_cell(c) for c in row.strip("|").split("|")]
        for row in table_lines[2:]
    ]

    num_rows = len(rows)
    num_cols = len(rows[0])

    collapsed_columns = 0

    for col_idx in range(num_cols):
        column = [r[col_idx] for r in rows]
        non_empty = [v for v in column if v not in ("", "-", "—")]
        empty_ratio = column.count("") / num_rows

        # ✅ values exist but most rows empty → inherited merge
        if non_empty and empty_ratio >= 0.5:
            collapsed_columns += 1

    return collapsed_columns >= 2


# -----------------------------
# BROKEN TABLE DECISION (FIXED)
# -----------------------------

def is_broken_table(
    docx_lines: List[str],
    pdf_lines: Optional[List[str]] = None
) -> bool:

    if len(docx_lines) < 2:
        return False

    header_cells = [normalize_cell(c) for c in docx_lines[0].strip("|").split("|")]
    header_cols = len(header_cells)
    meaningful_headers = sum(1 for c in header_cells if c not in ("", "-", "—"))

    # Header collapse
    if header_cols > 0 and meaningful_headers == 0:
        return True

    if header_cols > 0 and meaningful_headers / header_cols < 0.7:
        return True

    if has_vertical_merge_collapse(docx_lines):
        return True

    total_cells = 0
    meaningful_cells = 0

    for row in docx_lines[2:]:
        cells = [normalize_cell(c) for c in row.strip("|").split("|")]
        total_cells += len(cells)
        meaningful_cells += sum(1 for c in cells if c not in ("", "-", "—"))

        if len(cells) != header_cols:
            return True

    if total_cells > 0 and meaningful_cells == 0:
        return True

    if total_cells > 0 and meaningful_cells / total_cells < 0.1:
        return True

    if pdf_lines and count_columns(docx_lines[0]) < count_columns(pdf_lines[0]):
        return True

    if empty_cell_ratio(docx_lines) > 0.2:
        return True

    if has_repeated_cells(docx_lines):
        return True

    return False   # ✅ MISSING BEFORE — CRITICAL FIX


# -----------------------------
# Table matching
# -----------------------------

def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(len(a | b), 1)


def table_signature(block: Block):
    tokens = set(re.sub(r"[^\w\s]", "", block.lines[0].lower()).split())
    return {"heading": block.heading, "tokens": tokens, "cols": count_columns(block.lines[0])}


def best_pdf_table(docx_block, pdf_tables):
    sig = table_signature(docx_block)
    best, best_score = None, 0.0

    for pdf in pdf_tables:
        psig = table_signature(pdf)
        score = 0
        if sig["heading"] == psig["heading"]:
            score += 3
        score += jaccard(sig["tokens"], psig["tokens"]) * 2
        if sig["cols"] == psig["cols"]:
            score += 1

        if score > best_score:
            best_score = score
            best = pdf

    return best if best_score >= 1.5 else None   # ✅ relaxed threshold


# -----------------------------
# FORCE REPLACEMENT HELPERS
# -----------------------------

def is_strongly_broken_table(table_lines: List[str]) -> bool:
    body_cells = []
    for row in table_lines[2:]:
        body_cells.extend(normalize_cell(c) for c in row.strip("|").split("|"))

    meaningful = [c for c in body_cells if c not in ("", "-", "—")]

    if not meaningful:
        return True

    if len(meaningful) / max(len(body_cells), 1) < 0.05:
        return True

    return False


def fallback_pdf_table(docx_block, pdf_tables):
    docx_cols = count_columns(docx_block.lines[0])
    return min(
        pdf_tables,
        key=lambda p: abs(count_columns(p.lines[0]) - docx_cols),
        default=None
    )


# -----------------------------
# MAIN REPLACEMENT LOGIC (FIXED)
# -----------------------------

def replace_broken_tables(source_md: str, reference_md: str) -> str:
    source_blocks = parse_markdown_blocks(source_md)
    pdf_blocks = parse_markdown_blocks(reference_md)
    pdf_tables = [b for b in pdf_blocks if b.type == "table"]

    output: List[str] = []

    for block in source_blocks:
        if block.type != "table":
            output.extend(block.lines)
            continue

        candidate = best_pdf_table(block, pdf_tables)

        if is_strongly_broken_table(block.lines) and not candidate:
            candidate = fallback_pdf_table(block, pdf_tables)

        # ✅ FORCE really means FORCE now
        if candidate and (
            is_broken_table(block.lines, candidate.lines)
            or is_strongly_broken_table(block.lines)
        ):
            # output.append("<!-- TABLE_REPLACED_FROM_REFERENCE -->")
            output.extend(candidate.lines)
            # output.append("<!-- END_TABLE_REPLACED_FROM_REFERENCE -->")
        else:
            output.extend(block.lines)

    return "\n".join(output)


# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-md", required=True)
    parser.add_argument("--reference-md", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", default="merged.md")
    args = parser.parse_args()

    with open(args.source_md, encoding="utf-8") as f:
        source_md = f.read()

    with open(args.reference_md, encoding="utf-8") as f:
        reference_md = f.read()

    merged = replace_broken_tables(source_md, reference_md)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, args.output_name)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(merged)

    print(f"✅ Replaced Markdown saved to: {out_path}")


if __name__ == "__main__":
    main()