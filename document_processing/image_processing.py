import base64
import re
import logging
import shutil
import json
import hashlib
from pathlib import Path
from PIL import Image as PILImage
import numpy as np

from .vlm_interface import get_vlm_summary  

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

file_handler = logging.FileHandler("image_processing.log", encoding="utf-8")
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
MAX_CONTEXT_CHARS = 1500  # FIX for Issue 2

# ---------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)")
IMAGE_RE = re.compile(
    r'!\[[^\]]*\]\(\s*data:image/([^;]+);base64,([^)]*)\)',
    re.DOTALL
)

# ---------------------------------------------------------------------
# Image filtering
# ---------------------------------------------------------------------
def is_significant_image(image_path, min_width=80, min_height=80, min_size_kb=2):
    try:
        size_kb = image_path.stat().st_size / 1024
        if size_kb < min_size_kb:
            return False

        img = PILImage.open(image_path)
        w, h = img.size
        if w < min_width or h < min_height:
            return False

        arr = np.array(img.convert("RGB"))
        white_pixels = np.sum(
            (arr[:, :, 0] > 240) &
            (arr[:, :, 1] > 240) &
            (arr[:, :, 2] > 240)
        )
        total_pixels = arr.shape[0] * arr.shape[1]
        if (white_pixels / total_pixels) * 100 > 80:
            return False

        return True
    except Exception as e:
        logger.warning(f"Image check failed: {e}")
        return True


# ---------------------------------------------------------------------
# Markdown parsing
# ---------------------------------------------------------------------
def parse_sections(markdown):
    sections = []
    current = None

    for line in markdown.splitlines():
        m = HEADING_RE.match(line)
        if m:
            current = {
                "level": len(m.group(1)),
                "title": m.group(2).strip(),
                "lines": []
            }
            sections.append(current)
        else:
            if current is None:
                current = {
                    "level": 0,
                    "title": "Introduction",
                    "lines": []
                }
                sections.append(current)
            current["lines"].append(line)

    return sections


# ---------------------------------------------------------------------
# Split section into blocks
# ---------------------------------------------------------------------
def split_section_by_images(section):
    blocks, buffer = [], []

    for line in section["lines"]:
        if IMAGE_RE.search(line):
            if buffer:
                blocks.append({
                    "type": "text",
                    "content": "\n".join(buffer)
                })
                buffer = []
            blocks.append({"type": "image", "line": line})
        else:
            buffer.append(line)

    if buffer:
        blocks.append({
            "type": "text",
            "content": "\n".join(buffer)
        })

    return blocks


# ---------------------------------------------------------------------
# Group consecutive images (ISSUE 1 FIXED)
# ---------------------------------------------------------------------
def collect_image_groups(blocks):
    """
    NOTE (Intentional behavior):
    - Whitespace-only text blocks are NOT emitted.
    - They are skipped to prevent formatting noise from breaking image grouping.
    - This behavior is logged explicitly to avoid silent data loss.
    """
    grouped = []
    i = 0

    def is_nonblocking_text(b):
        return b["type"] == "text" and not b["content"].strip()

    while i < len(blocks):

        if blocks[i]["type"] == "text" and blocks[i]["content"].strip():
            grouped.append(blocks[i])
            i += 1
            continue

        image_lines = []
        start_index = i

        while i < len(blocks):
            if blocks[i]["type"] == "image":
                image_lines.append(blocks[i]["line"])
                i += 1
            elif is_nonblocking_text(blocks[i]):
                logger.debug(
                    "Skipping whitespace-only text block before/image group "
                    f"(index {i})"
                )
                i += 1
            else:
                break

        if image_lines:
            grouped.append({
                "type": "image_group",
                "lines": image_lines,
                "start_index": start_index
            })

    return grouped


# ---------------------------------------------------------------------
# Image context retrieval (ISSUE 2 FIXED)
# ---------------------------------------------------------------------
def get_image_context(blocks, image_index):
    for i in range(image_index - 1, -1, -1):
        if blocks[i]["type"] == "text" and blocks[i]["content"].strip():
            text = blocks[i]["content"].strip()
            if len(text) > MAX_CONTEXT_CHARS:
                logger.debug(
                    f"Context clipped from {len(text)} to {MAX_CONTEXT_CHARS} chars"
                )
            return text[-MAX_CONTEXT_CHARS:]
    return ""


# ---------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------
def process_embedded_images_in_markdown(md_content, output_dir):

    output_dir = Path(output_dir)
    extracted_dir = output_dir / "extracted_images"
    filtered_dir = output_dir / "filtered_images"

    extracted_dir.mkdir(parents=True, exist_ok=True)
    filtered_dir.mkdir(parents=True, exist_ok=True)

    IMAGE_REGISTRY = []
    final_lines = []
    img_counter = 0

    for section in parse_sections(md_content):

        if section["level"] > 0:
            final_lines.append("#" * section["level"] + " " + section["title"])
            final_lines.append("")

        raw_blocks = split_section_by_images(section)
        blocks = collect_image_groups(raw_blocks)

        for block in blocks:
            if block["type"] == "text":
                final_lines.append(block["content"])
                continue

            # ---------------- IMAGE GROUP ----------------
            context = get_image_context(raw_blocks, block["start_index"])
            image_bytes, image_names = [], []

            for line in block["lines"]:
                m = IMAGE_RE.search(line)
                if not m:
                    continue

                ext, b64 = m.groups()
                img_id = img_counter
                img_counter += 1

                data = base64.b64decode(b64)
                name = f"image_{img_id}.{ext}"
                path = extracted_dir / name
                path.write_bytes(data)

                if is_significant_image(path):
                    shutil.copy2(path, filtered_dir / name)
                    image_bytes.append(data)
                    image_names.append(name)

            if not image_bytes:
                continue

            # ISSUE 4 FIX — Stable ID
            hash_input = (section["title"] + context + ",".join(image_names)).encode()
            group_hash = hashlib.sha1(hash_input).hexdigest()[:10]
            group_id = f"image_group_{group_hash}"

            summary = get_vlm_summary(
                image_bytes,
                            context_text=f"""
            Section: {section['title']}

            {context}

            These images are consecutive and represent a connected UI or flow.
            Return STRICT JSON with keys: caption, description, ui_elements.
            """
            )
        # TO SKIP THE VLM SUMMARY STEP (for testing), you can use this hardcoded summary:
            # summary = {
            #     "caption": f"Summary for {', '.join(image_names)}",
            #     "description": f"Description for {', '.join(image_names)}", 
            #     "ui_elements": []
            # }

            IMAGE_REGISTRY.append({
                "image_group_id": group_id,
                "section": section["title"],
                "image_ids": image_names,
                "caption": summary.get("caption", ""),
                "description": summary.get("description", ""),
                "ui_elements": summary.get("ui_elements", []),
                "context": context
            })

            final_lines.append(
                f"[IMAGE_GROUP: {group_id}]\n\n"
                f"**Caption:** {summary.get('caption','')}\n\n"
                f"**Description:** {summary.get('description','')}"
            )

    with open(output_dir / "image_metadata.json", "w", encoding="utf-8") as f:
        json.dump(IMAGE_REGISTRY, f, indent=2, ensure_ascii=False)

    return "\n".join(final_lines)

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Section-aware markdown image processor")
    parser.add_argument("md_file")
    parser.add_argument("output_dir")

    args = parser.parse_args()

    with open(args.md_file, "r", encoding="utf-8") as f:
        md_content = f.read()

    result = process_embedded_images_in_markdown(md_content, args.output_dir)
    output_md_path = Path(args.output_dir) / "output.md"
    with open(output_md_path, "w", encoding="utf-8") as out_md:
        out_md.write(result)
    logger.info(f"Markdown output saved to {output_md_path}")