import sys
import re
import base64
import logging
from pathlib import Path
from docling.document_converter import DocumentConverter,PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc.base import ImageRefMode
import requests

from document_processing.tables_processing import replace_broken_tables
from document_processing.image_processing import process_embedded_images_in_markdown
from document_processing.chunking_process import parse_md_to_chunks, load_image_metadata, save_chunks

# Setup logging
log_file = Path("./extraction_debug.log")
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # Also print to console
    ]
)
logger = logging.getLogger(__name__)

def smart_chunk_markdown(md_content, output_folder):
    """
    Perform smart chunking: split by headers, and handle tables separately.
    """
    chunks_dir = output_folder / 'chunks'
    chunks_dir.mkdir(parents=True, exist_ok=True)
    
    lines = md_content.split('\n')
    chunks = []
    current_chunk = []
    
    for line in lines:
        if re.match(r'^#{1,4}\s', line):  # Header line
            if current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
        current_chunk.append(line)
    
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    # Now, for each chunk, check for tables and split further
    final_chunks = []
    for chunk in chunks:
        if '|' in chunk and '\n|' in chunk:  # Likely has table
            # Split by table
            parts = re.split(r'(\n\|.*\|(?:\n\|.*\|)*\n)', chunk)
            for part in parts:
                if part.strip():
                    final_chunks.append(part.strip())
        else:
            final_chunks.append(chunk)
    
    # Save chunks
    for i, chunk in enumerate(final_chunks):
        with open(chunks_dir / f'chunk_{i}.txt', 'w', encoding='utf-8') as f:
            f.write(chunk)


def main(input_file, output_folder):
    """
    Main function to process DOCX or PDF.
    """
    input_path = Path(input_file)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        logger.error(f"Input file {input_file} does not exist.")
        return

    # Determine format
    if input_path.suffix.lower() == '.pdf':
        converter = DocumentConverter(
                 format_options={
                     InputFormat.PDF: PdfFormatOption(
                         pipeline_options=PdfPipelineOptions(
                             generate_picture_images=True,
                             do_table_structure=True,
                         )
                    ),
                 }
                    )
    else:
        converter = DocumentConverter()

    # Convert document
    logger.info(f"Converting {input_path.name}...")
    result = converter.convert(str(input_path))
    doc = result.document

    # --- Step 1: generate markdown with embedded base64 images ---
    logger.info("Generating Markdown with embedded images...")
    md_content = doc.export_to_markdown(image_mode=ImageRefMode.EMBEDDED)

    # --- Step 2: save the original markdown (no replacements) ---
    with open(output_path / 'document_original.md', 'w', encoding='utf-8') as f:
        f.write(md_content)
    logger.info(f"Saved original Markdown to {output_path / 'document_original.md'}")

    # Debug: report how many embedded images were found
    embedded_count = len(re.findall(r'!\[[^\]]*\]\(data:image/', md_content))
    logger.info(f"Found {embedded_count} embedded base64 image(s) in markdown")

    # # --- Step 3: process images from markdown, replace with summaries ---
    # logger.info("Processing embedded images in markdown...")
    # md_with_summaries = process_embedded_images_in_markdown(md_content, output_path)

    # # --- Step 4: save the replaced markdown ---
    # with open(output_path / 'document.md', 'w', encoding='utf-8') as f:
    #     f.write(md_with_summaries)
    # logger.info(f"Saved Markdown with summaries to {output_path / 'document.md'}")

    # # --- Step 5: chunk the replaced markdown ---
    # logger.info("Chunking document...")
    # smart_chunk_markdown(md_with_summaries, output_path)

    logger.info(f"Processing complete. Output in {output_folder}")
    logger.info(f"Log file: {log_file}")



# -----------------------------
# Main replacement function
# -----------------------------

def main_dual(docx_file, pdf_file, output_folder):
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- DOCX ---
    logger.info(f"Converting DOCX: {docx_file}")
    docx_converter = DocumentConverter()
    docx_result = docx_converter.convert(str(docx_file))
    docx_doc = docx_result.document
    docx_md = docx_doc.export_to_markdown(image_mode=ImageRefMode.EMBEDDED)
    with open(output_path / 'document_docx_original.md', 'w', encoding='utf-8') as f:
        f.write(docx_md)

    # --- PDF ---
    logger.info(f"Converting PDF: {pdf_file}")
    pdf_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=PdfPipelineOptions(
                    generate_picture_images=True,
                    do_table_structure=True,
                )
            ),
        }
    )
    pdf_result = pdf_converter.convert(str(pdf_file))
    pdf_doc = pdf_result.document
    pdf_md = pdf_doc.export_to_markdown(image_mode=ImageRefMode.EMBEDDED)
    with open(output_path / 'document_pdf_original.md', 'w', encoding='utf-8') as f:
        f.write(pdf_md)

    # --- Replace broken tables in DOCX md with PDF tables ---
    logger.info("Replacing broken tables in DOCX-based Markdown with PDF-based tables...")
    merged_md = replace_broken_tables(docx_md, pdf_md)
    with open(output_path / 'document.md', 'w', encoding='utf-8') as f:
        f.write(merged_md)

    # --- Continue with image processing ---
    logger.info("Processing embedded images in markdown...")
    md_with_summaries = process_embedded_images_in_markdown(merged_md, output_path)
    with open(output_path / 'document_with_summaries.md', 'w', encoding='utf-8') as f:
        f.write(md_with_summaries)
    
    # --- Load image metadata and chunk with image linking ---
    logger.info("Chunking document...")
    image_metadata = load_image_metadata(str(output_path))
    if image_metadata:
        logger.info(f"[OK] Loaded {len(image_metadata)} image groups for linking")
    else:
        logger.warning("[!] No image metadata found")
    
    # Create chunks directory
    chunks_dir = output_path / 'chunks'
    chunks_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse and save chunks with image linking
    chunks = parse_md_to_chunks(md_with_summaries, str(output_path), image_metadata)
    save_chunks(chunks, str(chunks_dir))
    
    logger.info(f"[OK] Generated {len(chunks)} chunks with {sum(1 for c in chunks if c['metadata'].get('has_images'))} linked to images")
    logger.info(f"Processing complete. Output in {output_folder}")
    logger.info(f"Log file: {log_file}")


if __name__ == "__main__":
    if len(sys.argv) == 4:
        docx_file = sys.argv[1]
        pdf_file = sys.argv[2]
        output_folder = sys.argv[3]
        main_dual(docx_file, pdf_file, output_folder)
    elif len(sys.argv) == 3:
        input_file = sys.argv[1]
        output_folder = sys.argv[2]
        main(input_file, output_folder)
    else:
        logger.error("Usage: python process_doc.py <docx_file> <pdf_file> <output_folder> OR python process_doc.py <input_file> <output_folder>")
        sys.exit(1)