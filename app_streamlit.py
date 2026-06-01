"""
Streamlit RAG UI for SRS Chunks
- Retrieval from Qdrant
- Local LLM answering from Ollama
- Model selection
- Chunk display
- Conversation history with past interactions
"""

import streamlit as st
import requests
import json
from datetime import datetime
from typing import List, Dict
from pathlib import Path
from retrieval.retrieval_pipeline import SRSRetrievalPipeline, SearchConfig, SearchStrategy

# Page config
st.set_page_config(
    page_title="SRS RAG System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
OLLAMA_URL = "http://10.117.100.61:11434"
QDRANT_URL = "http://10.188.105.70:6333"

# =========================================================================
# UTILITY FUNCTIONS
# =========================================================================

@st.cache_data
def get_available_models():
    """Fetch available models from Ollama"""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        response.raise_for_status()
        models = response.json().get("models", [])
        return [m.get("name", "").replace(":latest", "") for m in models if m.get("name")]
    except Exception as e:
        st.error(f"❌ Could not fetch models from Ollama: {e}")
        return ["mxbai-embed-large"]

def get_llm_response(question: str, context: str, model: str) -> str:
    """Get response from local LLM"""
    try:
        prompt = f"""You are an SRS (Software Requirements Specification) expert. 
Answer the user's question based on the provided context from the SRS document only.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
        # Ensure model name format is correct
        model_name = model if ":" in model else f"{model}:latest"
        
        # Prepare request payload
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False
        }
        
        # Make request to Ollama
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            timeout=120
        )
        
        # Handle error responses
        if response.status_code != 200:
            error_msg = response.text if response.text else f"HTTP {response.status_code}"
            return f"❌ Error from Ollama: {error_msg}"
        
        result = response.json()
        return result.get("response", "No response generated")
        
    except requests.exceptions.Timeout:
        return "❌ Request timed out. The model might be busy or not responding."
    except requests.exceptions.ConnectionError:
        return f"❌ Cannot connect to Ollama at {OLLAMA_URL}. Is it running?"
    except Exception as e:
        return f"❌ Error getting LLM response: {str(e)}"

def resolve_image_path(img_path: str) -> str:
    """
    Resolve image path from workspace root or current directory.
    """
    img_path = str(img_path).strip()
    
    # Get workspace root (parent of app directory)
    app_dir = Path(__file__).parent
    workspace_root = app_dir.parent
    
    # Try from workspace root first (most common case)
    test_path = workspace_root / img_path
    if test_path.exists():
        return str(test_path)
    
    # Try as-is (in case it's already absolute or cwd-relative)
    test_path = Path(img_path)
    if test_path.exists():
        return str(test_path)
    
    return None

# =========================================================================
# SIDEBAR CONFIGURATION
# =========================================================================

with st.sidebar:
    st.title("⚙️ Configuration")
    
    # Model selection with refresh button
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Model Settings")
    with col2:
        if st.button("🔄", help="Refresh available models", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    available_models = get_available_models()
    
    selected_model = st.selectbox(
        "Select LLM Model",
        available_models,
        index=0 if available_models else 0,
        help="Choose which model to use for answering questions"
    )
    
    # Search strategy
    st.subheader("Search Strategy")
    search_strategy = st.radio(
        "Choose search strategy",
        options=[
            SearchStrategy.DENSE_ONLY,
            SearchStrategy.SPARSE_ONLY,
            SearchStrategy.HYBRID_RRF,
            SearchStrategy.HYBRID_WEIGHTED,
        ],
        index=2,
        format_func=lambda x: {
            SearchStrategy.DENSE_ONLY: "Dense (Semantic)",
            SearchStrategy.SPARSE_ONLY: "Sparse (Keywords)",
            SearchStrategy.HYBRID_RRF: "Hybrid Score-Aware (Recommended)",
            SearchStrategy.HYBRID_WEIGHTED: "Hybrid Weighted",
        }[x],
        help="Dense: Semantic/cosine similarity | Sparse: BM25 keyword match | Hybrid Score-Aware: Normalized fusion (50/50) | Hybrid Weighted: Custom weights"
    )
    
    # Results per page
    top_k = st.slider(
        "Number of chunks to retrieve",
        min_value=1,
        max_value=20,
        value=5,
        help="How many relevant chunks to use as context"
    )
    
    # Hybrid weighting (if selected)
    if search_strategy == SearchStrategy.HYBRID_WEIGHTED:
        st.subheader("Hybrid Weights")
        dense_weight = st.slider(
            "Dense weight (semantic)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1
        )
        sparse_weight = 1.0 - dense_weight
        st.metric("Sparse weight (keyword)", f"{sparse_weight:.1f}")
    else:
        dense_weight = 0.5
        sparse_weight = 0.5
    
    # Re-ranking settings
    st.markdown("---")
    st.subheader("🔄 Re-ranking")
    rerank_enabled = st.checkbox(
        "Enable Cross-Encoder Re-ranking",
        value=False,
        help="Use bge-reranker-large to re-score and re-rank results (adds latency)"
    )
    
    if rerank_enabled:
        st.info("ℹ️ Re-ranking will re-score each result with bge-reranker-large. Adds ~2-5 sec per result.")
        rerank_model = st.text_input(
            "Re-ranker Model",
            value="bge-reranker-large",
            help="Model name available in Ollama"
        )
    else:
        rerank_model = "bge-reranker-large"
    
    # System info
    st.markdown("---")
    st.subheader("📊 System Info")
    col1, col2 = st.columns(2)
    with col1:
        st.text("Ollama:")
        st.caption(f"🔗 {OLLAMA_URL}")
    with col2:
        st.text("Qdrant:")
        st.caption(f"🔗 {QDRANT_URL}")

# =========================================================================
# MAIN AREA
# =========================================================================

st.title("🔍 SRS RAG System")
st.markdown("**Retrieval-Augmented Generation for Technical Documentation**")

# Initialize session state
if "retrieval_pipeline" not in st.session_state:
    with st.spinner("Initializing retrieval pipeline..."):
        st.session_state.retrieval_pipeline = SRSRetrievalPipeline()

if "last_question" not in st.session_state:
    st.session_state.last_question = None
if "last_results" not in st.session_state:
    st.session_state.last_results = None
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "selected_history_idx" not in st.session_state:
    st.session_state.selected_history_idx = None

pipeline = st.session_state.retrieval_pipeline

# User question input
col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    user_question = st.text_input(
        "Ask your question about the SRS document:",
        placeholder="e.g., Explain the paper exit offset mechanism...",
        label_visibility="collapsed"
    )
with col2:
    search_button = st.button("🔍 Search", use_container_width=True)
with col3:
    clear_button = st.button("🗑️ Clear", use_container_width=True)

# Clear results if clear button is clicked
if clear_button:
    st.session_state.last_question = None
    st.session_state.last_results = None
    st.session_state.last_answer = None
    st.rerun()

# Process user question when search button is clicked
if search_button and user_question:
    # Step 1: Retrieve relevant chunks
    spinner_text = f"Searching with {search_strategy.value} strategy..."
    if rerank_enabled:
        spinner_text += " (+ re-ranking)"
    
    with st.spinner(spinner_text):
        config = SearchConfig(
            strategy=search_strategy,
            top_k=top_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            rerank_enabled=rerank_enabled,
            rerank_model=rerank_model
        )
        
        results = pipeline.search(user_question, config)
    
    if results:
        # Prepare context from retrieved chunks
        context = "\n\n".join([
            f"[Chunk {r.chunk_id}]\n{r.text}"
            for r in results
        ])
        
        with st.spinner(f"Generating answer using {selected_model}..."):
            answer = get_llm_response(user_question, context, selected_model)
        
        # Store in session state
        st.session_state.last_question = user_question
        st.session_state.last_results = results
        st.session_state.last_answer = answer
        
        # Add to conversation history
        conversation_entry = {
            "timestamp": datetime.now().isoformat(),
            "question": user_question,
            "answer": answer,
            "model": selected_model,
            "search_strategy": search_strategy.value,
            "results": results,
            "num_chunks": len(results)
        }
        st.session_state.conversation_history.append(conversation_entry)
        st.session_state.selected_history_idx = None  # Reset selection
    else:
        st.warning("⚠️ No relevant chunks found. Try a different query or search strategy.")

# Display stored results and conversation history in tabs
if st.session_state.last_question is not None or st.session_state.conversation_history:
    st.markdown("---")
    
    # Create tabs for current and history
    tab_labels = ["💬 Current Conversation"]
    if st.session_state.conversation_history:
        tab_labels.append(f"📜 History ({len(st.session_state.conversation_history)})")
    
    tabs = st.tabs(tab_labels)
    
    # ===== TAB 1: Current Conversation =====
    with tabs[0]:
        if st.session_state.last_question is not None:
            # Display current conversation
            results = st.session_state.last_results
            
            # Step 1: Display Retrieved Chunks
            st.subheader("📚 Retrieved Chunks")
            st.success(f"✅ Found {len(results)} relevant chunks")
            
            # Create tabs for each chunk (show rerank score in label if available)
            chunk_tab_labels = []
            for r in results:
                if r.rerank_score is not None:
                    label = f"Chunk {r.chunk_id} (Rerank: {r.rerank_score:.3f})"
                else:
                    label = f"Chunk {r.chunk_id} (Score: {r.score:.3f})"
                chunk_tab_labels.append(label)
            
            chunk_tabs = st.tabs(chunk_tab_labels)
            
            for tab, result in zip(chunk_tabs, results):
                with tab:
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        st.metric("Search Score", f"{result.score:.4f}")
                        if result.rerank_score is not None:
                            st.metric("Re-rank Score", f"{result.rerank_score:.4f}")
                        st.metric("Method", result.retrieval_method)
                        st.metric("Rank", result.rank)
                    
                    with col2:
                        st.subheader(f"Chunk {result.chunk_id}")
                        
                        # Metadata
                        with st.expander("📋 Metadata"):
                            for key, value in result.metadata.items():
                                if value:
                                    st.caption(f"**{key}:** {value}")
                        
                        # Images (if present)
                        if result.has_images:
                            with st.expander("📸 Associated Images (Debug Info)"):
                                st.write(f"**Has Images:** {result.has_images}")
                                # st.write(f"**Image Groups Type:** {type(result.image_groups)}")
                                # st.write(f"**Image Groups:** {result.image_groups}")
                                
                                if result.image_groups and isinstance(result.image_groups, list):
                                    for idx, img_group in enumerate(result.image_groups):
                                        st.write(f"\n**--- Image Group {idx} ---**")
                                        st.write(f"Type: {type(img_group)}")
                                        st.write(f"Data: {img_group}")
                                        
                                        if isinstance(img_group, dict):
                                            st.markdown(f"**🖼️ {img_group.get('image_group_id', 'Image Group')}**")
                                            st.caption(f"Caption: {img_group.get('caption', 'N/A')}")
                                            st.caption(f"Description: {img_group.get('description', 'N/A')}")
                                            
                                            image_paths = img_group.get("image_paths", [])
                                            # st.write(f"Image Paths Type: {type(image_paths)}")
                                            # st.write(f"Image Paths: {image_paths}")
                                            
                                            if image_paths:
                                                if isinstance(image_paths, str):
                                                    # Handle case where paths are stored as string
                                                    image_paths = [p.strip() for p in image_paths.split(",")]
                                                
                                                st.markdown("**Images:**")
                                                for img_path in image_paths:
                                                    img_path = str(img_path).strip()
                                                    resolved_path = resolve_image_path(img_path)
                                                    
                                                    if resolved_path:
                                                        try:
                                                            st.image(resolved_path, width="content")
                                                            st.caption(f"✅ {img_path}")
                                                        except Exception as e:
                                                            st.error(f"Error displaying image: {e}")
                                                    else:
                                                        st.error(f"❌ Image not found: {img_path}")
                                            
                                            if img_group.get("ui_elements"):
                                                st.caption(f"UI Elements: {', '.join(img_group['ui_elements'])}")
                                else:
                                    st.warning("Image groups data is not in expected format")
                        
                        # Content
                        st.subheader("Content")
                        st.markdown(result.text)
            
            # Step 2: Display LLM Response
            st.markdown("---")
            st.subheader("🤖 LLM Response")
            
            # Display answer
            st.markdown("### Answer:")
            st.info(st.session_state.last_answer)
            
            # Download response
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Markdown format
                # Build chunks section
                chunks_section = "\n".join([
                    f"### Chunk {r.chunk_id}\n**Search Score:** {r.score:.4f}" + 
                    (f"\n**Re-rank Score:** {r.rerank_score:.4f}" if r.rerank_score is not None else "") + 
                    f"\n{r.text}" 
                    for r in results
                ])
                
                markdown_content = f"""# SRS RAG Query Response

## Question
{st.session_state.last_question}

## Answer
{st.session_state.last_answer}

## Metadata
- **Model Used:** {selected_model}
- **Search Strategy:** {search_strategy.value}
- **Chunks Retrieved:** {len(results)}
- **Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Retrieved Chunks
{chunks_section}
"""
                st.download_button(
                    label="📄 Download as Markdown",
                    data=markdown_content,
                    file_name=f"srs_answer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            
            with col2:
                # JSON format
                json_content = json.dumps({
                    "question": st.session_state.last_question,
                    "answer": st.session_state.last_answer,
                    "model": selected_model,
                    "search_strategy": search_strategy.value,
                    "chunks_retrieved": len(results),
                    "timestamp": datetime.now().isoformat(),
                    "chunks": [
                        {
                            "id": r.chunk_id,
                            "score": float(r.score),
                            "rerank_score": float(r.rerank_score) if r.rerank_score is not None else None,
                            "method": r.retrieval_method,
                            "text": r.text,
                            "metadata": r.metadata,
                            "has_images": r.has_images,
                            "image_groups": r.image_groups if r.image_groups else []
                        }
                        for r in results
                    ]
                }, indent=2)
                st.download_button(
                    label="📋 Download as JSON",
                    data=json_content,
                    file_name=f"srs_answer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            # Advanced details
            with st.expander("📊 Search Details"):
                context = "\n\n".join([
                    f"[Chunk {r.chunk_id}]\n{r.text}"
                    for r in results
                ])
                st.write(f"**Search Strategy:** {search_strategy.value}")
                st.write(f"**Model Used:** {selected_model}")
                st.write(f"**Chunks Retrieved:** {len(results)}")
                st.write(f"**Total Context Length:** {len(context)} characters")
        else:
            st.info("💬 No current conversation. Ask a question to start.")
    
    # ===== TAB 2: Conversation History =====
    if len(tabs) > 1:
        with tabs[1]:
            if st.session_state.conversation_history:
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.subheader("📜 Past Conversations")
                with col2:
                    if st.button("🔄 Refresh", use_container_width=True):
                        st.rerun()
                with col3:
                    if st.button("🗑️ Clear All", use_container_width=True, key="clear_history"):
                        st.session_state.conversation_history = []
                        st.session_state.selected_history_idx = None
                        st.rerun()
                
                st.markdown("---")
                
                # Display conversation history
                for idx, conv in enumerate(reversed(st.session_state.conversation_history)):
                    rev_idx = len(st.session_state.conversation_history) - 1 - idx
                    timestamp = datetime.fromisoformat(conv["timestamp"]).strftime('%Y-%m-%d %H:%M:%S')
                    
                    with st.expander(f"**Q:** {conv['question'][:60]}... | {timestamp}", expanded=False):
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            # Display conversation details
                            st.markdown(f"**Question:** {conv['question']}")
                            st.markdown(f"**Model:** `{conv['model']}` | **Strategy:** `{conv['search_strategy']}`")
                            st.markdown(f"**Chunks Retrieved:** {conv['num_chunks']}")
                            
                            st.markdown("**Answer:**")
                            st.info(conv['answer'])
                        
                        with col2:
                            # Action buttons
                            col_inner1, col_inner2 = st.columns([1, 1])
                            
                            with col_inner1:
                                if st.button("📂 Load", key=f"load_{rev_idx}", use_container_width=True):
                                    st.session_state.last_question = conv['question']
                                    st.session_state.last_results = conv['results']
                                    st.session_state.last_answer = conv['answer']
                                    st.session_state.selected_history_idx = rev_idx
                                    st.rerun()
                            
                            with col_inner2:
                                if st.button("🗑️ Delete", key=f"delete_{rev_idx}", use_container_width=True):
                                    st.session_state.conversation_history.pop(rev_idx)
                                    st.rerun()
            else:
                st.info("📜 No conversation history yet. Start chatting to build history!")

else:
    # Default view
    st.markdown("""
    ## How to use:
    
    1. **Configure** your preferences on the left sidebar (model, search strategy, etc.)
    2. **Enter** your question in the input field above
    3. **Click** the Search button or press Enter
    4. **Review** the retrieved chunks and see the LLM's answer
    
    ## Features:
    
    - 🔍 **Multiple Search Strategies**: Semantic, Keyword, and Hybrid search
    - 🤖 **Local LLM**: Uses Ollama for private, on-premise processing
    - 📚 **Chunk Retrieval**: Shows which parts of the SRS are being used
    - 💾 **Download Results**: Save answers as Markdown or JSON
    - 📜 **Conversation History**: Review and reload past conversations
    - 🔄 **Model Refresh**: Update available models without restarting
    
    ## Getting Started:
    
    Simply type your question about the SRS document and click Search!
    """)
