"""
Streamlit interface for Multimodal RAG Query System
"""

import streamlit as st
from query import query_gemini, SOURCE_TEXTS, SOURCE_IMAGES, ParsedResponse
import textwrap
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="The Batch RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .answer-box {
        background-color: rgba(28, 131, 225, 0.1);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
        color: inherit;
    }
    .source-box {
        background-color: rgba(128, 128, 128, 0.1);
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid rgba(128, 128, 128, 0.3);
        margin: 0.5rem 0;
    }
    .source-box strong {
        color: #1f77b4;
    }
    .source-box a {
        color: #4dabf7;
    }
    .source-box p {
        color: inherit !important;
    }
    .image-caption {
        font-size: 0.9rem;
        color: #666;
        font-style: italic;
        margin-top: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'current_query' not in st.session_state:
    st.session_state.current_query = ""

# Sidebar
with st.sidebar:
    st.image("https://www.deeplearning.ai/wp-content/uploads/2021/02/LogoFiles_V3_Download-150x150.png", width=100)
    st.title("Settings")
    
    k_results = st.slider(
        "Number of articles to retrieve",
        min_value=5,
        max_value=20,
        value=10,
        help="More articles provide better context but slower responses"
    )
    
    st.divider()
    
    st.subheader("About")
    st.info("""
    This RAG system searches through **The Batch** newsletter archives 
    to answer your questions using both text and images.
    
    Powered by:
    - 🔍 ChromaDB + CLIP embeddings
    - 🤖 Google Gemini AI
    - 📰 The Batch by DeepLearning.AI
    """)
    
    st.divider()
    
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# Main content
st.markdown('<p class="main-header">🤖 The Batch RAG Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Ask questions about AI news from The Batch newsletter</p>', unsafe_allow_html=True)

# Example questions
with st.expander("💡 Example Questions"):
    col1, col2 = st.columns(2)
    with col1:
        if st.button("What are recent developments in multimodal AI?"):
            st.session_state.current_query = "What are recent developments in multimodal AI?"
            st.rerun()
        if st.button("How do chatbots influence people?"):
            st.session_state.current_query = "How do chatbots influence people?"
            st.rerun()
    with col2:
        if st.button("What are the latest trends in computer vision?"):
            st.session_state.current_query = "What are the latest trends in computer vision?"
            st.rerun()
        if st.button("Tell me about AI safety concerns"):
            st.session_state.current_query = "Tell me about AI safety concerns"
            st.rerun()

# Query input
query_text = st.text_input(
    "Your Question:",
    value=st.session_state.current_query,
    placeholder="e.g., What are the latest developments in large language models?",
    key="query_input"
)

col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    search_button = st.button("🔍 Search", type="primary", use_container_width=True)
with col2:
    clear_button = st.button("Clear", use_container_width=True)

if clear_button:
    st.session_state.current_query = ""
    st.rerun()

# Process query
if search_button and query_text:
    with st.spinner("🔍 Searching through The Batch archives..."):
        try:
            # Query the system
            parsed_response, (text_results, image_data) = query_gemini(query_text, k=k_results)
            print(parsed_response)
            
            # Store in history
            st.session_state.history.insert(0, {
                'query': query_text,
                'answer': parsed_response.response,
                'images': parsed_response.images,
                'links': parsed_response.links,
                'sources': text_results,
                'image_data': image_data
            })
            
            # Clear current query
            st.session_state.current_query = ""
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Display results
if st.session_state.history:
    for idx, item in enumerate(st.session_state.history):
        st.divider()
        
        # Query
        st.markdown(f"### 🔎 Query: {item['query']}")
        
        # Answer
        st.markdown("#### 💬 Answer")
        st.markdown(f'<div class="answer-box">{item["answer"]}</div>', unsafe_allow_html=True)
        
        # Links from answer (if any)
        if item.get('links'):
            st.markdown("#### 🔗 Related Links")
            for link in item['links']:
                st.markdown(f"- [{link}]({link})")
        
        # Images from answer
        if item['images']:
            st.markdown("#### 🖼️ Relevant Images")
            
            # Display in columns
            num_images = len(item['images'])
            cols = st.columns(min(3, num_images))
            
            for i, img_url in enumerate(item['images']):
                with cols[i % 3]:
                    try:
                        st.image(img_url, use_container_width=True)
                        st.caption(f"Source: {img_url}")
                    except:
                        st.warning(f"Could not load image: {img_url}")
        
        # Sources
        with st.expander(f"📚 View {len(item['sources'])} Sources"):
            for source in item['sources']:
                st.markdown(f"""
                <div class="source-box">
                    <strong>{source['title']}</strong> ({source['date']})<br>
                    <a href="{source['url']}" target="_blank">{source['url']}</a><br>
                    <p style="margin-top: 0.5rem;">{textwrap.shorten(source['text'], width=300)}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Additional images
        if item['image_data']:
            with st.expander(f"🎨 View {len(item['image_data'])} Additional Images from Articles"):
                img_cols = st.columns(3)
                for i, (img_path, caption) in enumerate(item['image_data']):
                    with img_cols[i % 3]:
                        try:
                            st.image(img_path, use_container_width=True)
                            if caption:
                                st.caption(caption)
                        except:
                            st.warning(f"Could not load: {img_path}")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>Powered by ChromaDB, CLIP, and Google Gemini | Data from The Batch by DeepLearning.AI</p>
</div>
""", unsafe_allow_html=True)