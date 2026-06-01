import os
import re
import yaml
import uuid
import requests
import glob
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, SparseVectorParams, Distance

# Try to import fastembed for sparse embeddings
try:
    from fastembed import SparseTextEmbedding
    HAS_SPARSE = True
except ImportError:
    HAS_SPARSE = False
    print("⚠️  Warning: fastembed not installed. Sparse embeddings will be disabled.")
    print("   Install with: pip install fastembed")


# =========================
# CONFIG
# =========================

QDRANT_URL = "http://10.188.105.70:6333"
OLLAMA_URL = "http://10.117.100.61:11434"
COLLECTION_NAME = "srs_chunks_image"
EMBEDDING_MODEL = "mxbai-embed-large"
SPARSE_MODEL_NAME = "Qdrant/bm25"  # For sparse embeddings
# TODO upgradable to new sparse_model
# SPARSE_MODEL_NAME = "naver/splade-cocondenser-ensembledistil"

# Chunk directories to ingest
CHUNK_DIRS = [
    ".JEA-04135/document_output/chunks"
]

BATCH_SIZE = 32  # Process embeddings in batches


# =========================
# INIT QDRANT CLIENT & SPARSE MODEL
# =========================

client = QdrantClient(QDRANT_URL)

# Initialize sparse model if available
sparse_model = None
if HAS_SPARSE:
    try:
        sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)
        print(f"✅ Sparse model loaded: {SPARSE_MODEL_NAME}")
    except Exception as e:
        print(f"⚠️  Could not load sparse model: {e}")


# =========================
# CREATE COLLECTION
# =========================

def create_collection(vector_size: int = 1024):
    """Create Qdrant collection for storing hybrid (dense + sparse) embeddings."""
    try:
        # Try to delete existing collection
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"✅ Deleted existing collection: {COLLECTION_NAME}")
        except Exception:
            pass

        # Create collection with both dense and sparse vectors
        vectors_config = {
            "dense": VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        }

        # Add sparse vectors config if sparse model is available
        sparse_vectors_config = None
        if HAS_SPARSE and sparse_model:
            sparse_vectors_config = {
                "sparse": SparseVectorParams()
            }
            print(f"🔄 Creating collection with HYBRID retrieval (dense + sparse)...")
        else:
            print(f"📌 Creating collection with DENSE vectors only...")

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config
        )
        
        if sparse_vectors_config:
            print(f"✅ Created hybrid collection: {COLLECTION_NAME} (dense + sparse)")
        else:
            print(f"✅ Created collection: {COLLECTION_NAME} (dense only)")
    except Exception as e:
        print(f"❌ Error creating collection: {e}")
        raise


# =========================
# OLLAMA EMBEDDING FUNCTION
# =========================

def get_ollama_embedding(text: str) -> Optional[List[float]]:
    """Get embedding from Ollama server."""
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json={
                "model": EMBEDDING_MODEL,
                "input": text
            },
            timeout=60
        )
        response.raise_for_status()
        embedding = response.json().get("embeddings", [[]])[0]
        return embedding if embedding else None
    except requests.exceptions.RequestException as e:
        print(f"❌ Error getting embedding from Ollama: {e}")
        return None


def batch_get_ollama_embeddings(texts: List[str]) -> List[Optional[List[float]]]:
    """Get embeddings for multiple texts efficiently."""
    embeddings = []
    for text in tqdm(texts, desc="Getting dense embeddings", unit="text"):
        embedding = get_ollama_embedding(text)
        embeddings.append(embedding)
    return embeddings


# =========================
# SPARSE EMBEDDING FUNCTION
# =========================

def get_sparse_embedding(text: str) -> Optional[Dict]:
    """Get sparse embedding from fastembed."""
    if not HAS_SPARSE or not sparse_model:
        return None
    
    try:
        sparse_vec = list(sparse_model.embed(text))[0]
        return {
            "indices": sparse_vec.indices.tolist() if hasattr(sparse_vec.indices, 'tolist') else list(sparse_vec.indices),
            "values": sparse_vec.values.tolist() if hasattr(sparse_vec.values, 'tolist') else list(sparse_vec.values)
        }
    except Exception as e:
        print(f"⚠️  Error getting sparse embedding: {e}")
        return None


def batch_get_sparse_embeddings(texts: List[str]) -> List[Optional[Dict]]:
    """Get sparse embeddings for multiple texts efficiently."""
    if not HAS_SPARSE or not sparse_model:
        return [None] * len(texts)
    
    sparse_embeddings = []
    for text in tqdm(texts, desc="Getting sparse embeddings", unit="text"):
        sparse_emb = get_sparse_embedding(text)
        sparse_embeddings.append(sparse_emb)
    return sparse_embeddings


# =========================
# PARSE CHUNK FILE
# =========================

def parse_chunk_file(file_path: str) -> Tuple[Optional[Dict], Optional[str]]:
    """Parse a single chunk file (YAML metadata + content)."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split by first '---' separator (metadata end marker)
        parts = content.split('---', 2)
        if len(parts) < 3:
            print(f"⚠️ Invalid format in {file_path}")
            return None, None

        # parts[0] is empty (before first ---)
        # parts[1] is YAML metadata
        # parts[2] is content
        metadata_str = parts[1].strip()
        chunk_content = parts[2].strip()

        metadata = yaml.safe_load(metadata_str)
        
        if not metadata or not chunk_content:
            return None, None
            
        return metadata, chunk_content
    except Exception as e:
        print(f"❌ Error parsing {file_path}: {e}")
        return None, None


# =========================
# CLEAN + STRUCTURE TEXT
# =========================

def normalize_text(content: str) -> str:
    """Clean and normalize text for embedding."""
    # Remove extra markdown formatting but keep structure
    content = re.sub(r'\*{2,}', '', content)  # Remove bold **
    content = re.sub(r'_{2,}', '', content)   # Remove underline __
    content = re.sub(r'~{2,}', '', content)   # Remove strikethrough ~~
    
    # Normalize whitespace
    content = re.sub(r'\n{3,}', '\n\n', content)  # Max 2 newlines
    content = re.sub(r'[ \t]+\n', '\n', content)  # Remove trailing spaces
    
    return content.strip()


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract important keywords from text."""
    # Extract words that look like identifiers, codes, or concepts
    keywords = re.findall(r'\b[A-Z][A-Za-z0-9\-_]*\b|\b[0-9]+-[0-9]+\b', text)
    # Remove duplicates and limit
    return list(set(keywords))[:max_keywords]


# =========================
# BUILD QDRANT POINTS
# =========================

def build_point(metadata: Dict, content: str, embedding: List[float], sparse_embedding: Optional[Dict] = None) -> Optional[PointStruct]:
    """Build a single Qdrant point from metadata, content, and embeddings (dense + sparse)."""
    try:
        normalized_content = normalize_text(content)

        # Create enriched text for better semantic search (uses normalized version)
        enriched_text = f"[{metadata.get('feature_id', 'N/A')}] {normalized_content}"

        # Extract image metadata from chunk
        has_images = metadata.get("has_images", False)
        image_groups = metadata.get("image_groups", [])
        
        # Build image metadata for payload
        image_metadata = []
        if image_groups:
            for img_group in image_groups:
                image_metadata.append({
                    "image_group_id": img_group.get("image_group_id") or img_group.get("id"),
                    "caption": img_group.get("caption", ""),
                    "description": img_group.get("description", ""),
                    "image_ids": img_group.get("image_ids", []),
                    "image_paths": img_group.get("image_paths", []),
                    "ui_elements": img_group.get("ui_elements", [])
                })

        payload = {
            "doc_id": metadata.get("doc_id"),
            "product": metadata.get("product"),
            "feature_id": metadata.get("feature_id"),
            "section": metadata.get("section", ""),
            "subsection": metadata.get("subsection", ""),
            "chunk_type": metadata.get("chunk_type", "text"),
            "version": metadata.get("version"),
            "source": metadata.get("source"),
            "chunk_id": metadata.get("chunk_id"),
            "text": content,  # ✅ STORE ORIGINAL TEXT (not normalized)
            "text_normalized": normalized_content,  # Store normalized version for reference
            "keywords": extract_keywords(content),
            "enriched_text": enriched_text,
            # ✅ NEW: Image metadata
            "has_images": has_images,
            "image_groups": image_metadata
        }

        # Build vector dict with dense vector
        vectors = {
            "dense": embedding
        }
        
        # Add sparse vector if available
        if sparse_embedding:
            vectors["sparse"] = sparse_embedding

        point = PointStruct(
            id=uuid.uuid4().int % (2**63),  # Use integer ID
            vector=vectors,
            payload=payload
        )
        return point
    except Exception as e:
        print(f"❌ Error building point: {e}")
        return None


# =========================
# DISCOVER AND INGEST CHUNKS
# =========================

def discover_chunk_files(chunk_dirs: Optional[List[str]] = None) -> List[str]:
    """Discover all chunk files in configured directories."""
    if chunk_dirs is None:
        chunk_dirs = CHUNK_DIRS
    
    chunk_files = []
    
    for chunk_dir in chunk_dirs:
        if os.path.exists(chunk_dir):
            files = glob.glob(os.path.join(chunk_dir, "chunk_*.txt"))
            chunk_files.extend(files)
            print(f"📁 Found {len(files)} chunks in {chunk_dir}")
        else:
            print(f"⚠️  Chunk directory not found: {chunk_dir}")
    
    return sorted(chunk_files)


def ingest_chunks(chunk_files: List[str]) -> int:
    """Ingest all chunk files into Qdrant with hybrid (dense + sparse) embeddings."""
    if not chunk_files:
        print("❌ No chunk files found!")
        return 0

    print(f"\n📊 Starting ingestion of {len(chunk_files)} chunks...")

    # Step 1: Parse all chunks
    print("\n📝 Parsing chunk files...")
    parsed_chunks = []
    for file_path in tqdm(chunk_files, desc="Parsing", unit="file"):
        metadata, content = parse_chunk_file(file_path)
        if metadata and content:
            parsed_chunks.append((metadata, content, file_path))

    print(f"✅ Successfully parsed {len(parsed_chunks)} chunks")

    if not parsed_chunks:
        print("❌ No valid chunks to ingest!")
        return 0

    # Step 2: Extract texts for embedding
    texts_to_embed = [content for _, content, _ in parsed_chunks]

    # Step 3: Get dense embeddings from Ollama
    print(f"\n🔗 Getting dense embeddings from Ollama ({OLLAMA_URL})...")
    dense_embeddings = batch_get_ollama_embeddings(texts_to_embed)

    # Check if all embeddings were successful
    failed_dense = sum(1 for e in dense_embeddings if e is None)
    if failed_dense > 0:
        print(f"⚠️ Failed to get {failed_dense} dense embeddings")

    # Step 4: Get sparse embeddings (if available)
    sparse_embeddings = None
    if HAS_SPARSE and sparse_model:
        print(f"\n📊 Getting sparse embeddings (BM25)...")
        sparse_embeddings = batch_get_sparse_embeddings(texts_to_embed)
        failed_sparse = sum(1 for e in sparse_embeddings if e is None)
        if failed_sparse > 0:
            print(f"⚠️ Failed to get {failed_sparse} sparse embeddings")
    else:
        sparse_embeddings = [None] * len(texts_to_embed)
        print(f"\n⚙️ Sparse embeddings disabled (fastembed not available)")

    # Step 5: Build points with both embeddings
    print("\n🔨 Building Qdrant points (hybrid vectors)...")
    points = []
    for (metadata, content, _), dense_emb, sparse_emb in tqdm(
        zip(parsed_chunks, dense_embeddings, sparse_embeddings),
        total=len(parsed_chunks),
        desc="Building",
        unit="point"
    ):
        if dense_emb:  # Only create point if dense embedding was successful
            point = build_point(metadata, content, dense_emb, sparse_emb)
            if point:
                points.append(point)

    print(f"✅ Built {len(points)} points")

    if not points:
        print("❌ Failed to build any points!")
        return 0

    # Step 6: Upload to Qdrant
    print(f"\n⬆️  Uploading {len(points)} points to Qdrant...")
    try:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        
        # Count hybrid vectors and image statistics
        hybrid_count = sum(1 for p in points if "sparse" in p.vector)
        dense_only_count = len(points) - hybrid_count
        image_count = sum(1 for p in points if p.payload.get("has_images"))
        total_image_groups = sum(len(p.payload.get("image_groups", [])) for p in points)
        
        print(f"✅ Successfully uploaded {len(points)} points to Qdrant!")
        print(f"   • Hybrid (dense + sparse): {hybrid_count}")
        print(f"   • Dense only: {dense_only_count}")
        print(f"   • 📸 Chunks with images: {image_count}/{len(points)}")
        print(f"   • 🖼️  Total image groups linked: {total_image_groups}")
        
        return len(points)
    except Exception as e:
        print(f"❌ Error uploading to Qdrant: {e}")
        return 0


# =========================
# HEALTH CHECK
# =========================

def check_ollama_health() -> bool:
    """Check if Ollama server is available."""
    try:
        response = requests.get(
            f"{OLLAMA_URL}/api/tags",
            timeout=5
        )
        if response.status_code == 200:
            print(f"✅ Ollama server is healthy at {OLLAMA_URL}")
            return True
    except Exception as e:
        print(f"❌ Cannot reach Ollama server at {OLLAMA_URL}: {e}")
        return False


def check_qdrant_health() -> bool:
    """Check if Qdrant server is available."""
    try:
        client.get_collections()
        print(f"✅ Qdrant server is healthy at {QDRANT_URL}")
        return True
    except Exception as e:
        print(f"❌ Cannot reach Qdrant server at {QDRANT_URL}: {e}")
        return False


# =========================
# MAIN
# =========================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest SRS chunks into Qdrant using Ollama embeddings"
    )
    parser.add_argument(
        "--chunk-dir",
        type=str,
        action="append",
        dest="chunk_dirs",
        help="Path to chunk directory (can be specified multiple times)"
    )
    parser.add_argument(
        "--create-collection",
        action="store_true",
        help="Create/recreate the collection before ingestion"
    )
    parser.add_argument(
        "--vector-size",
        type=int,
        default=1024,
        help="Size of embedding vectors (default: 1024)"
    )
    parser.add_argument(
        "--skip-health-check",
        action="store_true",
        help="Skip health checks for servers"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("SRS CHUNK INGESTION - Qdrant + Ollama + BM25")
    print("HYBRID RETRIEVAL (Dense + Sparse Vectors)")
    print("=" * 60)

    # Health checks
    if not args.skip_health_check:
        print("\n🏥 Checking server health...")
        if not check_qdrant_health():
            return
        if not check_ollama_health():
            return

    # Create collection if requested
    if args.create_collection:
        print("\n📦 Creating collection...")
        create_collection(vector_size=args.vector_size)

    # Discover chunks
    print("\n🔍 Discovering chunk files...")
    chunk_files = discover_chunk_files(chunk_dirs=args.chunk_dirs)

    if not chunk_files:
        print("❌ No chunk files found in configured directories!")
        return

    # Ingest chunks
    ingested_count = ingest_chunks(chunk_files)

    print("\n" + "=" * 60)
    if ingested_count > 0:
        print(f"✅ SUCCESS: Ingested {ingested_count} chunks into Qdrant")
    else:
        print("❌ FAILED: No chunks were ingested")
    print("=" * 60)


if __name__ == "__main__":
    main()
