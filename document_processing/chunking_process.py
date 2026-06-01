import os
import re
import uuid
import json
import argparse
from typing import List, Dict
from pathlib import Path

# ---------- CONFIG ----------
DOC_ID = "JEA-04135"
PRODUCT = "L7.0"
FEATURE_ID = "EBX_RCR_1193"
VERSION = "1.05"

MAX_CHARS = 2800
MIN_CHARS = 800

# Regex to extract image group IDs
IMAGE_GROUP_PATTERN = re.compile(r'\[IMAGE_GROUP:\s*(image_group_\w+)\]')

# ---------- HELPERS ----------
def is_table(block: str) -> bool:
    return "|" in block and "---" in block


def load_image_metadata(base_dir: str) -> Dict:
    """Load image metadata from image_metadata.json if it exists"""
    metadata_path = os.path.join(base_dir, "image_metadata.json")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Could not load image metadata: {e}")
            return []
    return []


def extract_image_groups_from_text(text: str) -> List[str]:
    """Extract all image group IDs from text"""
    return IMAGE_GROUP_PATTERN.findall(text)


def get_image_paths(base_dir: str, image_names: List[str]) -> List[str]:
    """Get full paths to filtered images"""
    filtered_dir = os.path.join(base_dir, "filtered_images")
    paths = []
    for img_name in image_names:
        path = os.path.join(filtered_dir, img_name)
        if os.path.exists(path):
            paths.append(path)
    return paths


def split_with_overlap(text: str, max_len: int, overlap: int) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_len
        chunk = text[start:end]
        chunks.append(chunk)
        start = max(end - overlap, 0)
        if start == end:
            break
    return chunks


# ---------- PARSER ----------

def parse_md_to_chunks(md_text: str, output_dir: str, image_metadata: List[Dict] = None) -> list:
    """Parse markdown to chunks and link associated images"""
    if image_metadata is None:
        image_metadata = []
    
    lines = md_text.split("\n")
    chunks = []

    current_section = ""
    current_subsection = ""
    current_subsubsection = ""
    buffer = []
    image_groups_in_chunk = []  # Track images in current chunk

    in_table = False

    def heading_block():
        h = []
        if current_section:
            h.append(f"## {current_section}")
        if current_subsection:
            h.append(f"### {current_subsection}")
        if current_subsubsection:
            h.append(f"#### {current_subsubsection}")
        return "\n".join(h)

    def flush_chunk(force=False):
        nonlocal buffer, image_groups_in_chunk
        content = "\n".join(buffer).strip()
        if not content:
            buffer = []
            image_groups_in_chunk = []
            return

        if not force and len(content) < MIN_CHARS:
            return

        full_content = heading_block()
        if full_content:
            full_content += "\n\n" + content
        else:
            full_content = content

        # Build image metadata for this chunk
        image_meta_list = []
        for image_group_id in image_groups_in_chunk:
            for img_meta in image_metadata:
                if img_meta.get("image_group_id") == image_group_id:
                    image_meta_list.append({
                        "image_group_id": img_meta.get("image_group_id"),
                        "caption": img_meta.get("caption"),
                        "description": img_meta.get("description"),
                        "image_ids": img_meta.get("image_ids", []),
                        "ui_elements": img_meta.get("ui_elements", []),
                        "image_paths": get_image_paths(output_dir, img_meta.get("image_ids", []))
                    })
                    break

        chunks.append({
            "id": str(uuid.uuid4()),
            "content": full_content,
            "metadata": {
                "doc_id": DOC_ID,
                "product": PRODUCT,
                "feature_id": FEATURE_ID,
                "section": current_section,
                "subsection": current_subsection,
                "subsubsection": current_subsubsection,
                "chunk_type": "table" if in_table else "text",
                "version": VERSION,
                "source": "SRS",
                "image_groups": image_meta_list,  # Link to images
                "has_images": len(image_meta_list) > 0
            }
        })

        buffer = []
        image_groups_in_chunk = []

    for line in lines:
        # ---------- SECTION (##) ----------
        if line.startswith("## "):
            flush_chunk(force=True)
            current_section = line.replace("##", "").strip()
            current_subsection = ""
            current_subsubsection = ""
            continue

        # ---------- SUBSECTION (###) ----------
        if line.startswith("### "):
            flush_chunk(force=True)
            current_subsection = line.replace("###", "").strip()
            current_subsubsection = ""
            continue

        # ---------- SUB-SUBSECTION (####) ----------
        if line.startswith("#### "):
            flush_chunk(force=True)
            current_subsubsection = line.replace("####", "").strip()
            continue

        # ---------- IMAGE GROUPS ----------
        image_ids = extract_image_groups_from_text(line)
        if image_ids:
            image_groups_in_chunk.extend(image_ids)

        # ---------- TABLE ----------
        if line.strip().startswith("|"):
            in_table = True
            buffer.append(line)
            continue

        if in_table:
            if line.strip() == "":
                in_table = False
                buffer.append(line)
                flush_chunk(force=True)
            else:
                buffer.append(line)
            continue

        # ---------- NORMAL TEXT ----------
        buffer.append(line)

        if len("\n".join(buffer)) > MAX_CHARS:
            flush_chunk(force=True)

    flush_chunk(force=True)
    return chunks



# ---------- FILE IO ----------
def save_chunks(chunks: list, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    for idx, chunk in enumerate(chunks, start=1):
        chunk_id = f"{idx:03d}"
        file_path = os.path.join(output_dir, f"chunk_{chunk_id}.txt")

        metadata = chunk["metadata"]

        # ---------- Build metadata header ----------
        image_groups_str = ""
        if metadata.get("image_groups"):
            image_group_lines = []
            for img_group in metadata["image_groups"]:
                image_group_lines.append(f"  - image_group_id: {img_group['image_group_id']}")
                image_group_lines.append(f"    caption: {img_group['caption']}")
                image_group_lines.append(f"    description: {img_group['description']}")
                image_group_lines.append(f"    image_ids: {', '.join(img_group['image_ids']) if isinstance(img_group['image_ids'], list) else img_group['image_ids']}")
                
                # Properly format image_paths list
                img_paths = img_group.get('image_paths', [])
                if isinstance(img_paths, list) and img_paths:
                    image_group_lines.append(f"    image_paths:")
                    for path in img_paths:
                        image_group_lines.append(f"      - {path}")
                else:
                    image_group_lines.append(f"    image_paths: []")
                
                if img_group.get("ui_elements"):
                    ui_elems = img_group['ui_elements'] if isinstance(img_group['ui_elements'], list) else []
                    image_group_lines.append(f"    ui_elements: {ui_elems}")
            image_groups_str = "image_groups:\n" + "\n".join(image_group_lines) + "\n"

        header = (
            "---\n"
            f"doc_id: {metadata['doc_id']}\n"
            f"product: {metadata['product']}\n"
            f"feature_id: {metadata['feature_id']}\n"
            f"section: {metadata['section']}\n"
            f"subsection: {metadata['subsection']}\n"
            f"chunk_type: {metadata['chunk_type']}\n"
            f"version: {metadata['version']}\n"
            f"source: {metadata['source']}\n"
            f"chunk_id: {chunk_id}\n"
            f"has_images: {metadata.get('has_images', False)}\n"
            + image_groups_str +
            "---\n\n"
        )

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(header)
            f.write(chunk["content"])

    print(f"[OK] Saved {len(chunks)} chunks with image metadata")


# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="SRS Markdown Chunker for RAG with Image Linking")
    parser.add_argument("md_file", help="Path to input .md file")
    parser.add_argument("output_dir", help="Directory to write chunks")
    parser.add_argument("--image-dir", help="Optional: Directory containing image_metadata.json (defaults to parent of output_dir)")

    args = parser.parse_args()

    # Determine image metadata directory
    if args.image_dir:
        image_dir = args.image_dir
    else:
        # Default to parent directory of output_dir (document_output parent)
        image_dir = os.path.dirname(args.output_dir)

    # Load image metadata
    image_metadata = load_image_metadata(image_dir)
    if image_metadata:
        print(f"✅ Loaded {len(image_metadata)} image groups from {image_dir}")
    else:
        print(f"⚠️ No image metadata found in {image_dir}")

    # Read markdown
    with open(args.md_file, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Parse chunks with image linking
    chunks = parse_md_to_chunks(md_text, output_dir=image_dir, image_metadata=image_metadata)

    # Save chunks
    save_chunks(chunks, args.output_dir)

    print(f"✅ Generated {len(chunks)} chunks in '{args.output_dir}'")
    print(f"   - Image groups linked: {sum(1 for c in chunks if c['metadata'].get('has_images'))}")


if __name__ == "__main__":
    main()
