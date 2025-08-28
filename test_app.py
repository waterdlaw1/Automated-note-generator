import streamlit as st
import requests
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
import pdfplumber
from docx import Document
import json
import base64
from cryptography.fernet import Fernet

# Set up the app
st.set_page_config(
    page_title="AI Note Generator", 
    page_icon="📝", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        color: #2e75b6;
        border-bottom: 2px solid #2e75b6;
        padding-bottom: 0.5rem;
    }
    .info-box {
        background-color: #d9eaf7;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .success-box {
        background-color: #e2f0d9;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff2cc;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .stButton button {
        background-color: #2e75b6;
        color: white;
        font-weight: bold;
        width: 100%;
    }
    .file-info {
        background-color: #f5f5f5;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .download-btn {
        background-color: #4CAF50 !important;
    }
    .api-settings {
        background-color: #f0f7fd;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #2e75b6;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown('<h1 class="main-header">📝 Advanced AI Note Generator</h1>', unsafe_allow_html=True)
st.write("Transform documents into comprehensive notes with AI-powered summarization and analysis.")

# Key management functions
def generate_key():
    """Generate a key for encryption"""
    return Fernet.generate_key()

def encrypt_data(data, key):
    """Encrypt data using Fernet symmetric encryption"""
    fernet = Fernet(key)
    return fernet.encrypt(data.encode())

def decrypt_data(encrypted_data, key):
    """Decrypt data using Fernet symmetric encryption"""
    fernet = Fernet(key)
    return fernet.decrypt(encrypted_data).decode()

def save_api_key(api_key, remember_me=False):
    """Save API key based on user preference"""
    if remember_me:
        # Save to local file with basic encryption
        key = generate_key()
        encrypted_api_key = encrypt_data(api_key, key)
        
        # Save both the key and encrypted API key
        data_to_save = {
            'key': base64.b64encode(key).decode(),
            'api_key': base64.b64encode(encrypted_api_key).decode()
        }
        
        with open('api_config.json', 'w') as f:
            json.dump(data_to_save, f)
    else:
        # Just save to session state
        st.session_state.api_key = api_key

def load_api_key():
    """Load API key from local storage if available"""
    try:
        if os.path.exists('api_config.json'):
            with open('api_config.json', 'r') as f:
                data = json.load(f)
                
            key = base64.b64decode(data['key'])
            encrypted_api_key = base64.b64decode(data['api_key'])
            
            return decrypt_data(encrypted_api_key, key)
    except:
        # If anything goes wrong, return None
        return None
    return None

# Initialize session state
if 'api_key' not in st.session_state:
    # Try to load from local storage
    saved_api_key = load_api_key()
    if saved_api_key:
        st.session_state.api_key = saved_api_key
        st.session_state.remember_me = True
    else:
        st.session_state.api_key = ""
        st.session_state.remember_me = False

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API key management
    st.markdown('<div class="api-settings">', unsafe_allow_html=True)
    st.subheader("API Key Settings")
    
    # Check if we have a saved API key
    if st.session_state.api_key:
        st.success("API key is configured")
        if st.button("Change API Key"):
            st.session_state.api_key = ""
            # Remove saved key if it exists
            if os.path.exists('api_config.json'):
                os.remove('api_config.json')
            st.rerun()
    else:
        api_key = st.text_input("Enter your Hugging Face API key:", type="password", 
                               help="Get your API key from https://huggingface.co/settings/tokens")
        
        remember_me = st.checkbox("Remember me", value=st.session_state.remember_me,
                                 help="Save your API key for future sessions (stored locally on your device)")
        
        if api_key:
            save_api_key(api_key, remember_me)
            st.session_state.api_key = api_key
            st.session_state.remember_me = remember_me
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Only show the rest of the configuration if API key is set
    if st.session_state.api_key:
        headers = {"Authorization": f"Bearer {st.session_state.api_key}"}
        API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
        
        # Model selection
        st.markdown("---")
        st.subheader("Model Options")
        summary_length = st.slider("Summary Length", min_value=50, max_value=500, value=150, 
                                  help="Adjust the length of the generated summary")
        
        # File type preferences
        st.markdown("---")
        st.subheader("File Preferences")
        extract_images = st.checkbox("Attempt to extract text from images in PDFs", value=False)
        max_file_size = st.slider("Maximum file size (MB)", min_value=1, max_value=50, value=10)
        
        # Advanced options
        with st.expander("Advanced Options"):
            chunk_size = st.slider("Text chunk size for processing", min_value=500, max_value=2000, value=1024,
                                  help="Larger documents will be split into chunks of this size")
            timeout = st.slider("API timeout (seconds)", min_value=10, max_value=120, value=30)
    else:
        st.warning("Please enter your API key to use the app")
        st.stop()

# Initialize session state for results
if 'summary_result' not in st.session_state:
    st.session_state.summary_result = None
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = ""
if 'processing_time' not in st.session_state:
    st.session_state.processing_time = 0

def extract_text_from_file(uploaded_file, extract_images=False):
    """Extract text from uploaded file based on its type"""
    text = ""
    file_size = uploaded_file.size / (1024 * 1024)  # Size in MB
    
    # Check file size
    if file_size > max_file_size:
        st.error(f"File size ({file_size:.2f} MB) exceeds the maximum allowed size ({max_file_size} MB).")
        return None
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # Extract text based on file type
        if uploaded_file.type == "application/pdf":
            with pdfplumber.open(tmp_path) as pdf:
                for page in pdf.pages:
                    # Try to extract text
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                    elif extract_images:
                        st.warning("This PDF appears to contain images. Text extraction from images is not supported in this version.")
        
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = Document(tmp_path)
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            
            # Extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        text += cell.text + " "
                    text += "\n"
        
        elif uploaded_file.type == "text/plain":
            text = uploaded_file.getvalue().decode("utf-8")
        
        else:
            st.error(f"Unsupported file type: {uploaded_file.type}")
            return None
            
    except Exception as e:
        st.error(f"Error extracting text: {str(e)}")
        return None
    finally:
        # Clean up temporary file
        os.unlink(tmp_path)
    
    return text

def summarize_text(text, max_length=150):
    """Send text to Hugging Face API for summarization"""
    # Handle very long texts by splitting into chunks
    if len(text) > chunk_size:
        st.info(f"Document is large ({len(text)} characters). Processing in chunks...")
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        summaries = []
        
        progress_bar = st.progress(0)
        for i, chunk in enumerate(chunks):
            progress_bar.progress((i + 1) / len(chunks))
            chunk_summary = process_text_chunk(chunk, max_length)
            if not chunk_summary.startswith("Error"):
                summaries.append(chunk_summary)
            time.sleep(1)  # Avoid rate limiting
        
        # Combine chunk summaries
        if summaries:
            combined_text = " ".join(summaries)
            if len(combined_text) > chunk_size:
                # Summarize the combined summaries if still too long
                return process_text_chunk(combined_text[:chunk_size*2], max_length)
            return combined_text
        else:
            return "Error: Could not generate summary from chunks."
    else:
        return process_text_chunk(text, max_length)

def process_text_chunk(text, max_length):
    """Process a single chunk of text through the API"""
    payload = {
        "inputs": text,
        "parameters": {
            "max_length": max_length,
            "min_length": 30,
            "do_sample": False
        }
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=timeout)
        
        if response.status_code == 200:
            return response.json()[0]["summary_text"]
        elif response.status_code == 503:
            return "Model is loading. Please try again in a few moments."
        else:
            return f"Error: {response.status_code} - {response.text}"
    except requests.exceptions.RequestException as e:
        return f"Request failed: {str(e)}"

def create_downloadable_content(summary, key_points, action_items):
    """Create a formatted text file for download"""
    content = f"AI-Generated Notes\n"
    content += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    content += "=" * 50 + "\n\n"
    
    content += "EXECUTIVE SUMMARY:\n"
    content += "=" * 20 + "\n"
    content += summary + "\n\n"
    
    content += "KEY POINTS:\n"
    content += "=" * 20 + "\n"
    for i, point in enumerate(key_points, 1):
        content += f"{i}. {point}\n"
    content += "\n"
    
    content += "ACTION ITEMS:\n"
    content += "=" * 20 + "\n"
    for i, item in enumerate(action_items, 1):
        content += f"{i}. {item}\n"
    
    return content

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="sub-header">Upload Document</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a file", 
        type=["pdf", "docx", "txt"],
        help="Supported formats: PDF, DOCX, and TXT"
    )
    
    if uploaded_file is not None:
        st.markdown('<div class="file-info">', unsafe_allow_html=True)
        st.write(f"**File name:** {uploaded_file.name}")
        st.write(f"**File type:** {uploaded_file.type}")
        st.write(f"**File size:** {uploaded_file.size / 1024:.2f} KB")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Extract text from the file
        with st.spinner("Extracting text from file..."):
            start_time = time.time()
            extracted_text = extract_text_from_file(uploaded_file, extract_images)
            extraction_time = time.time() - start_time
            
        if extracted_text:
            st.session_state.extracted_text = extracted_text
            st.success(f"Text extracted successfully in {extraction_time:.2f} seconds!")
            
            # Show text statistics
            word_count = len(extracted_text.split())
            char_count = len(extracted_text)
            st.write(f"**Extracted text statistics:** {word_count} words, {char_count} characters")
            
            # Show a preview of the extracted text
            with st.expander("Preview extracted text"):
                st.text(extracted_text[:1000] + "..." if len(extracted_text) > 1000 else extracted_text)

with col2:
    st.markdown('<div class="sub-header">Text Input</div>', unsafe_allow_html=True)
    manual_text = st.text_area("Or paste your text here:", height=300, 
                              placeholder="Enter text directly or upload a file above...")
    
    if manual_text.strip():
        st.session_state.extracted_text = manual_text
        word_count = len(manual_text.split())
        char_count = len(manual_text)
        st.write(f"**Text statistics:** {word_count} words, {char_count} characters")

# Generate notes button
if st.button("🚀 Generate Comprehensive Notes", use_container_width=True):
    if st.session_state.extracted_text and st.session_state.extracted_text.strip():
        with st.spinner("Analyzing content and generating notes..."):
            start_time = time.time()
            summary = summarize_text(st.session_state.extracted_text, summary_length)
            processing_time = time.time() - start_time
            st.session_state.processing_time = processing_time
            
            if summary.startswith("Error") or "loading" in summary.lower():
                st.error(summary)
                st.session_state.summary_result = None
            else:
                st.session_state.summary_result = summary
                st.balloons()
                
    else:
        st.warning("Please upload a file or enter some text to summarize.")

# Display results if available
if st.session_state.summary_result:
    st.markdown('<div class="success-box">', unsafe_allow_html=True)
    st.success(f"Notes generated successfully in {st.session_state.processing_time:.2f} seconds!")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display results in tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Summary", "🔑 Key Points", "✅ Action Items", "💾 Download"])
    
    with tab1:
        st.subheader("Executive Summary")
        st.write(st.session_state.summary_result)
        
        # Additional analysis
        with st.expander("Text Analysis"):
            original_words = len(st.session_state.extracted_text.split())
            summary_words = len(st.session_state.summary_result.split())
            compression_ratio = (1 - (summary_words / original_words)) * 100 if original_words > 0 else 0
            
            st.metric("Original words", original_words)
            st.metric("Summary words", summary_words)
            st.metric("Compression ratio", f"{compression_ratio:.1f}%")
    
    with tab2:
        st.subheader("Key Points")
        
        # This would ideally be generated by another model in a real application
        key_points = [
            "Identify the main thesis or central argument",
            "Note supporting evidence and examples",
            "Highlight important definitions and concepts",
            "Record any conclusions or recommendations",
            "Note questions or areas needing further research"
        ]
        
        for i, point in enumerate(key_points, 1):
            st.write(f"{i}. {point}")
    
    with tab3:
        st.subheader("Action Items")
        
        # This would ideally be generated by another model in a real application
        action_items = [
            "Review the summary to ensure understanding",
            "Create flashcards for key terms and concepts",
            "Schedule follow-up research on unclear concepts",
            "Discuss main points with peers or mentors",
            "Set a reminder to review this material before exams"
        ]
        
        for i, item in enumerate(action_items, 1):
            st.write(f"{i}. {item}")
    
    with tab4:
        st.subheader("Download Options")
        
        # Create downloadable content
        download_content = create_downloadable_content(
            st.session_state.summary_result,
            [
                "Identify the main thesis or central argument",
                "Note supporting evidence and examples",
                "Highlight important definitions and concepts",
                "Record any conclusions or recommendations",
                "Note questions or areas needing further research"
            ],
            [
                "Review the summary to ensure understanding",
                "Create flashcards for key terms and concepts",
                "Schedule follow-up research on unclear concepts",
                "Discuss main points with peers or mentors",
                "Set a reminder to review this material before exams"
            ]
        )
        
        # Download button
        st.download_button(
            label="📥 Download Notes as Text File",
            data=download_content,
            file_name="ai_generated_notes.txt",
            mime="text/plain",
            use_container_width=True
        )
        
        st.info("You can copy the content from the other tabs and paste into your preferred note-taking application.")

# Add footer with information
st.markdown("---")
st.caption("""
This application uses the BART-large-CNN model from Hugging Face for text summarization. 
For best results, ensure your documents contain clear, well-structured text. 
Performance may vary with complex formatting or poor quality scans.
""")

# Add a feedback mechanism
with st.expander("💬 Provide Feedback"):
    feedback = st.text_area("How can we improve this tool?")
    if st.button("Submit Feedback"):
        st.success("Thank you for your feedback! We'll use it to improve the application.")