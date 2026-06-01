"""
Hybrid Retrieval Pipeline for SRS Chunks
Performs semantic search using both dense (Ollama) and sparse (BM25) vectors
with advanced ranking and filtering capabilities.
"""

import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import requests
from enum import Enum

# Try to import fastembed for sparse embeddings
try:
    from fastembed import SparseTextEmbedding
    HAS_SPARSE = True
except ImportError:
    HAS_SPARSE = False


class SearchStrategy(Enum):
    """Search strategy options"""
    DENSE_ONLY = "dense"           # Only semantic search
    SPARSE_ONLY = "sparse"         # Only keyword search
    HYBRID_RRF = "hybrid_rrf"      # Hybrid with Reciprocal Rank Fusion
    HYBRID_WEIGHTED = "hybrid_weighted"  # Hybrid with weighted scoring


@dataclass
class SearchResult:
    """Single search result"""
    chunk_id: str
    text: str
    metadata: Dict
    score: float
    retrieval_method: str  # 'dense', 'sparse', or 'hybrid'
    rank: int
    has_images: bool = False  # Whether chunk has associated images
    image_groups: List[Dict] = None  # Image metadata
    rerank_score: Optional[float] = None  # Cross-encoder re-rank score


@dataclass
class SearchConfig:
    """Configuration for search"""
    strategy: SearchStrategy = SearchStrategy.HYBRID_RRF
    top_k: int = 10
    dense_weight: float = 0.5  # For weighted strategy
    sparse_weight: float = 0.5  # For weighted strategy
    filter_metadata: Optional[Dict] = None  # Metadata filters
    min_score: float = 0.0  # Minimum score threshold
    rerank_enabled: bool = False  # Enable cross-encoder re-ranking
    rerank_model: str = "bge-reranker-large"  # Cross-encoder model name


class SRSRetrievalPipeline:
    """Hybrid retrieval pipeline for SRS chunks"""
    
    def __init__(
        self,
        qdrant_url: str = "http://10.188.105.70:6333",
        ollama_url: str = "http://10.117.100.61:11434",
        embedding_model: str = "mxbai-embed-large",
        sparse_model: str = "Qdrant/bm25",
        collection_name: str = "srs_chunks_image",
    ):
        """Initialize retrieval pipeline"""
        self.qdrant_url = qdrant_url
        self.ollama_url = ollama_url
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        
        # Initialize sparse model
        self.sparse_model = None
        if HAS_SPARSE:
            try:
                self.sparse_model = SparseTextEmbedding(model_name=sparse_model)
                print(f"✅ Sparse model loaded: {sparse_model}")
            except Exception as e:
                print(f"⚠️  Could not load sparse model: {e}")
        
        print(f"✅ Retrieval pipeline initialized")
    
    # =====================================================================
    # EMBEDDING FUNCTIONS
    # =====================================================================
    
    def _get_dense_embedding(self, text: str) -> Optional[List[float]]:
        """Get dense embedding from Ollama"""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/embed",
                json={
                    "model": self.embedding_model,
                    "input": text
                },
                timeout=30
            )
            response.raise_for_status()
            embedding = response.json().get("embeddings", [[]])[0]
            return embedding if embedding else None
        except Exception as e:
            print(f"❌ Error getting dense embedding: {e}")
            return None
    
    def _get_sparse_embedding(self, text: str) -> Optional[Dict]:
        """Get sparse embedding from fastembed"""
        if not self.sparse_model:
            return None
        
        try:
            sparse_vec = list(self.sparse_model.embed(text))[0]
            return {
                "indices": sparse_vec.indices.tolist() if hasattr(sparse_vec.indices, 'tolist') else list(sparse_vec.indices),
                "values": sparse_vec.values.tolist() if hasattr(sparse_vec.values, 'tolist') else list(sparse_vec.values)
            }
        except Exception as e:
            print(f"❌ Error getting sparse embedding: {e}")
            return None
    
    # =====================================================================
    # SEARCH FUNCTIONS
    # =====================================================================
    
    def _dense_search(self, query_embedding: List[float], config: SearchConfig) -> List[Tuple[int, float, Dict]]:
        """Perform dense vector search"""
        try:
            # Use HTTP API for searching with named vector
            response = requests.post(
                f"{self.qdrant_url}/collections/{self.collection_name}/points/search",
                json={
                    "vector": {
                        "name": "dense",
                        "vector": query_embedding
                    },
                    "limit": config.top_k * 2,
                    "with_payload": True
                },
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            results = []
            for result in data.get("result", []):
                score = result.get("score", 0)
                if score >= config.min_score:
                    results.append((
                        result.get("id"),
                        score,
                        result.get("payload", {})
                    ))
            return results
        except Exception as e:
            print(f"❌ Dense search error: {e}")
            return []
    
    def _sparse_search(self, query_embedding: Dict, config: SearchConfig) -> List[Tuple[int, float, Dict]]:
        """Perform sparse vector search"""
        if not query_embedding:
            return []
        
        try:
            # Use HTTP API for searching with named vector
            response = requests.post(
                f"{self.qdrant_url}/collections/{self.collection_name}/points/search",
                json={
                    "vector": {
                        "name": "sparse",
                        "vector": query_embedding
                    },
                    "limit": config.top_k * 2,
                    "with_payload": True
                },
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            results = []
            for result in data.get("result", []):
                score = result.get("score", 0)
                if score >= config.min_score:
                    results.append((
                        result.get("id"),
                        score,
                        result.get("payload", {})
                    ))
            return results
        except Exception as e:
            print(f"❌ Sparse search error: {e}")
            return []
    
    # =====================================================================
    # RANKING & FUSION
    # =====================================================================
    
    def _rerank_with_cross_encoder(self, query: str, text: str, model: str = "bge-reranker-large") -> Optional[float]:
        """
        Re-rank a result using cross-encoder (bge-reranker-large) via Ollama.
        Returns a relevance score between 0.0 and 1.0.
        """
        try:
            # Limit text to 512 tokens for efficiency
            text_preview = text[:1000] if len(text) > 1000 else text
            
            # Craft prompt for the cross-encoder
            prompt = f"""Query: {query}
Passage: {text_preview}

Relevance score (0-100):"""
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.0
                },
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            response_text = result.get("response", "").strip()
            
            # Extract numeric score from response
            import re
            numbers = re.findall(r'\d+', response_text)
            if numbers:
                score = float(numbers[0]) / 100.0  # Normalize to 0.0-1.0
                return score
            return None
        except Exception as e:
            print(f"⚠️ Re-ranking error: {e}")
            return None
    
    def _rerank_results(self, query: str, results: List[SearchResult], config: SearchConfig) -> List[SearchResult]:
        """
        Re-rank search results using cross-encoder model.
        Updates rerank_score field and re-sorts results.
        """
        if not config.rerank_enabled or not results:
            return results
        
        print(f"\n🔄 Re-ranking {len(results)} results with {config.rerank_model}...")
        
        # Score each result
        for result in results:
            rerank_score = self._rerank_with_cross_encoder(query, result.text, config.rerank_model)
            result.rerank_score = rerank_score
            if rerank_score is not None:
                print(f"   {result.chunk_id}: {rerank_score:.4f}")
        
        # Sort by rerank score (descending), with None scores at the end
        valid_results = [r for r in results if r.rerank_score is not None]
        invalid_results = [r for r in results if r.rerank_score is None]
        
        valid_results.sort(key=lambda r: r.rerank_score, reverse=True)
        reranked = valid_results + invalid_results
        
        # Update ranks
        for idx, result in enumerate(reranked, 1):
            result.rank = idx
        
        print(f"✅ Re-ranking complete")
        return reranked
    
    def _min_max_normalize(
        self,
        results: List[Tuple[int, float, Dict]]
    ) -> List[Tuple[int, float, Dict]]:
        """
        Min-max normalize scores to [0, 1] range.
        Handles empty lists and edge cases.
        """
        if not results:
            return []
        
        scores = [score for _, score, _ in results]
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score
        
        # Avoid division by zero
        if score_range == 0:
            return [(doc_id, 0.5, payload) for doc_id, _, payload in results]
        
        normalized = []
        for doc_id, score, payload in results:
            norm_score = (score - min_score) / score_range
            normalized.append((doc_id, norm_score, payload))
        
        return normalized
    
    def _score_aware_fusion(
        self,
        dense_results: List[Tuple[int, float, Dict]],
        sparse_results: List[Tuple[int, float, Dict]],
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5
    ) -> Dict[int, float]:
        """
        Combine dense and sparse results using score-aware fusion.
        Properly normalizes scores on different scales before combining.
        
        Process:
        1. Normalize each set of scores to [0, 1] using min-max scaling
        2. Weight by importance (dense_weight, sparse_weight)
        3. Sum weighted scores
        """
        fused_scores = {}
        
        # Normalize dense results
        if dense_results:
            normalized_dense = self._min_max_normalize(dense_results)
            for doc_id, norm_score, _ in normalized_dense:
                fused_scores[doc_id] = fused_scores.get(doc_id, 0) + (norm_score * dense_weight)
        
        # Normalize sparse results
        if sparse_results:
            normalized_sparse = self._min_max_normalize(sparse_results)
            for doc_id, norm_score, _ in normalized_sparse:
                fused_scores[doc_id] = fused_scores.get(doc_id, 0) + (norm_score * sparse_weight)
        
        return fused_scores
    
    
    def _weighted_fusion(
        self,
        dense_results: List[Tuple[int, float, Dict]],
        sparse_results: List[Tuple[int, float, Dict]],
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5
    ) -> Dict[int, float]:
        """Combine results using weighted scoring"""
        weighted_scores = {}
        
        # Normalize and score dense results
        if dense_results:
            max_dense_score = max(score for _, score, _ in dense_results) if dense_results else 1.0
            for doc_id, score, _ in dense_results:
                normalized = score / max_dense_score if max_dense_score > 0 else 0
                weighted_scores[doc_id] = weighted_scores.get(doc_id, 0) + (normalized * dense_weight)
        
        # Normalize and score sparse results
        if sparse_results:
            max_sparse_score = max(score for _, score, _ in sparse_results) if sparse_results else 1.0
            for doc_id, score, _ in sparse_results:
                normalized = score / max_sparse_score if max_sparse_score > 0 else 0
                weighted_scores[doc_id] = weighted_scores.get(doc_id, 0) + (normalized * sparse_weight)
        
        return weighted_scores
    
    # =====================================================================
    # MAIN SEARCH METHOD
    # =====================================================================
    
    def search(
        self,
        query: str,
        config: SearchConfig = None
    ) -> List[SearchResult]:
        """
        Perform hybrid search on SRS chunks.
        
        Args:
            query: Search query text
            config: SearchConfig object (uses defaults if None)
        
        Returns:
            List of SearchResult objects sorted by score
        """
        if config is None:
            config = SearchConfig()
        
        print(f"\n" + "=" * 70)
        print(f"🔍 HYBRID SEARCH")
        print(f"=" * 70)
        print(f"Query: {query[:60]}{'...' if len(query) > 60 else ''}")
        print(f"Strategy: {config.strategy.value}")
        print(f"Top K: {config.top_k}")
        
        results = []
        
        # Strategy: Dense only
        if config.strategy == SearchStrategy.DENSE_ONLY:
            print(f"\n📊 Dense Search Only...")
            dense_emb = self._get_dense_embedding(query)
            if dense_emb:
                dense_results = self._dense_search(dense_emb, config)
                for rank, (doc_id, score, payload) in enumerate(dense_results[:config.top_k], 1):
                    results.append(SearchResult(
                        chunk_id=payload.get("chunk_id"),
                        text=payload.get("text", ""),
                        metadata={k: v for k, v in payload.items() if k not in ["text", "has_images", "image_groups"]},
                        score=score,
                        retrieval_method="dense",
                        rank=rank,
                        has_images=payload.get("has_images", False),
                        image_groups=payload.get("image_groups", None)
                    ))
        
        # Strategy: Sparse only
        elif config.strategy == SearchStrategy.SPARSE_ONLY:
            print(f"\n📊 Sparse Search Only...")
            sparse_emb = self._get_sparse_embedding(query)
            if sparse_emb:
                sparse_results = self._sparse_search(sparse_emb, config)
                for rank, (doc_id, score, payload) in enumerate(sparse_results[:config.top_k], 1):
                    results.append(SearchResult(
                        chunk_id=payload.get("chunk_id"),
                        text=payload.get("text", ""),
                        metadata={k: v for k, v in payload.items() if k not in ["text", "has_images", "image_groups"]},
                        score=score,
                        retrieval_method="sparse",
                        rank=rank,
                        has_images=payload.get("has_images", False),
                        image_groups=payload.get("image_groups", None)
                    ))
        
        # Strategy: Hybrid with Score-Aware Fusion
        elif config.strategy == SearchStrategy.HYBRID_RRF:
            print(f"\n📊 Hybrid Search (Score-Aware Fusion)...")
            
            dense_emb = self._get_dense_embedding(query)
            sparse_emb = self._get_sparse_embedding(query)
            
            dense_results = self._dense_search(dense_emb, config) if dense_emb else []
            sparse_results = self._sparse_search(sparse_emb, config) if sparse_emb else []
            
            print(f"   Dense results: {len(dense_results)}")
            print(f"   Sparse results: {len(sparse_results)}")
            
            if not dense_results and not sparse_results:
                return []
            
            # Fuse results with proper score normalization
            fused_scores = self._score_aware_fusion(
                dense_results,
                sparse_results,
                dense_weight=0.5,
                sparse_weight=0.5
            )
            
            # Build results from all sources
            all_docs = {}
            for doc_id, score, payload in dense_results + sparse_results:
                if doc_id not in all_docs:
                    all_docs[doc_id] = payload
            
            # Sort by fused score
            sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
            
            for rank, (doc_id, score) in enumerate(sorted_docs[:config.top_k], 1):
                payload = all_docs.get(doc_id, {})
                results.append(SearchResult(
                    chunk_id=payload.get("chunk_id"),
                    text=payload.get("text", ""),
                    metadata={k: v for k, v in payload.items() if k not in ["text", "has_images", "image_groups"]},
                    score=score,
                    retrieval_method="hybrid (score-aware)",
                    rank=rank,
                    has_images=payload.get("has_images", False),
                    image_groups=payload.get("image_groups", None)
                ))
        
        # Strategy: Hybrid with weighted fusion
        elif config.strategy == SearchStrategy.HYBRID_WEIGHTED:
            print(f"\n📊 Hybrid Search (Weighted)...")
            
            dense_emb = self._get_dense_embedding(query)
            sparse_emb = self._get_sparse_embedding(query)
            
            dense_results = self._dense_search(dense_emb, config) if dense_emb else []
            sparse_results = self._sparse_search(sparse_emb, config) if sparse_emb else []
            
            print(f"   Dense results: {len(dense_results)}")
            print(f"   Sparse results: {len(sparse_results)}")
            
            if not dense_results and not sparse_results:
                return []
            
            # Fuse results
            fused_scores = self._weighted_fusion(
                dense_results,
                sparse_results,
                config.dense_weight,
                config.sparse_weight
            )
            
            # Build results from all sources
            all_docs = {}
            for doc_id, score, payload in dense_results + sparse_results:
                if doc_id not in all_docs:
                    all_docs[doc_id] = payload
            
            # Sort by fused score
            sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
            
            for rank, (doc_id, score) in enumerate(sorted_docs[:config.top_k], 1):
                payload = all_docs.get(doc_id, {})
                results.append(SearchResult(
                    chunk_id=payload.get("chunk_id"),
                    text=payload.get("text", ""),
                    metadata={k: v for k, v in payload.items() if k not in ["text", "has_images", "image_groups"]},
                    score=score,
                    retrieval_method="hybrid (weighted)",
                    rank=rank,
                    has_images=payload.get("has_images", False),
                    image_groups=payload.get("image_groups", None)
                ))
        
        print(f"\n✅ Found {len(results)} results")
        
        # Apply re-ranking if enabled
        if config.rerank_enabled and results:
            results = self._rerank_results(query, results, config)
        
        return results
    
    # =====================================================================
    # UTILITY FUNCTIONS
    # =====================================================================
    
    def format_results(self, results: List[SearchResult]) -> str:
        """Format search results for display"""
        if not results:
            return "No results found"
        
        output = []
        output.append("\n" + "=" * 70)
        output.append("SEARCH RESULTS")
        output.append("=" * 70)
        
        for result in results:
            # Display base score and retrieval method
            score_str = f"SCORE: {result.score:.4f}"
            
            # Add re-rank score if available
            if result.rerank_score is not None:
                score_str += f" | RERANK: {result.rerank_score:.4f}"
            
            output.append(f"\n[#{result.rank}] {score_str} | METHOD: {result.retrieval_method}")
            output.append(f"Chunk ID: {result.chunk_id}")
            for key, value in result.metadata.items():
                if key not in ["keywords", "enriched_text"]:
                    output.append(f"  {key}: {value}")
            output.append(f"\nText Preview:\n{result.text}\n")
            output.append("-" * 70)
        
        return "\n".join(output)
    
    def search_and_print(
        self,
        query: str,
        config: SearchConfig = None
    ) -> List[SearchResult]:
        """Search and print results"""
        results = self.search(query, config)
        print(self.format_results(results))
        return results


# =========================================================================
# EXAMPLE USAGE
# =========================================================================

if __name__ == "__main__":
    # Initialize pipeline
    pipeline = SRSRetrievalPipeline()
    
    # Example queries
    queries = [
        "explain paper exit offset position"
    ]
    
    print("\n" + "=" * 70)
    print("RETRIEVAL PIPELINE DEMO")
    print("=" * 70)
    
    for query in queries:
        print(f"\n\n{'#' * 70}")
        print(f"QUERY: {query}")
        print(f"{'#' * 70}")
        
        # Test different strategies
        strategies = [
            SearchStrategy.DENSE_ONLY,
            SearchStrategy.SPARSE_ONLY,
            SearchStrategy.HYBRID_RRF,
        ]
        
        for strategy in strategies:
            config = SearchConfig(
                strategy=strategy,
                top_k=3
            )
            
            try:
                results = pipeline.search_and_print(query, config)
            except Exception as e:
                print(f"❌ Error with {strategy.value}: {e}")
