import streamlit as st
import requests
from PIL import Image
import io
import base64
import time

# Configure the page
st.set_page_config(
    page_title="Jaundice Diagnosis Tool",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .upload-section {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .result-section {
        background-color: #e8f5e8;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .error-section {
        background-color: #ffe6e6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 Jaundice Diagnosis Tool</h1>', unsafe_allow_html=True)
    
    # Sidebar for server configuration
    with st.sidebar:
        st.header("⚙️ Server Configuration")
        server_url = st.text_input(
            "Server URL",
            value="https://112dac6022d0.ngrok-free.app",  # Replace with your actual ngrok URL
            help="URL of your Flask server (use ngrok URL if running on Colab)"
        )
        
        # Test server connection
        if st.button("Test Connection"):
            try:
                response = requests.get(f"{server_url}/health")
                if response.status_code == 200:
                    st.success("✅ Server is running!")
                    st.info(f"Connected to: {server_url}")
                else:
                    st.warning("⚠️ Server responded but health endpoint not found")
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Cannot connect to server: {str(e)}")
                st.info("💡 If using Colab, make sure to use the ngrok URL")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📁 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload an image to analyze for jaundice"
        )
        
        if uploaded_file is not None:
            # Display image info
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / 1024:.1f} KB",
                "File type": uploaded_file.type
            }
            st.write("**File Details:**")
            for key, value in file_details.items():
                st.write(f"- {key}: {value}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.header("🖼️ Preview")
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Analyze button
                if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                    analyze_image(uploaded_file, server_url)
                    
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
        else:
            st.info("Upload an image to see preview")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Results section
    if 'analysis_result' in st.session_state:
        display_results()

def analyze_image(uploaded_file, server_url):
    """Analyze the uploaded image using the Flask server"""
    
    with st.spinner("🔬 Analyzing image..."):
        try:
            # Prepare the file for upload
            files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            
            # Make request to Flask server
            response = requests.post(f"{server_url}/predict", files=files, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                st.session_state.analysis_result = {
                    'success': True,
                    'prediction': result.get('prediction', 'Unknown'),
                    'confidence': result.get('confidence', 0.0),
                    'details': result.get('details', {})
                }
                st.success("✅ Analysis completed!")
                
            else:
                st.session_state.analysis_result = {
                    'success': False,
                    'error': f"Server error: {response.status_code}",
                    'details': response.text
                }
                st.error(f"❌ Analysis failed: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            st.session_state.analysis_result = {
                'success': False,
                'error': "Cannot connect to server. Make sure the Flask server is running.",
                'details': {}
            }
            st.error("❌ Cannot connect to server")
            
        except requests.exceptions.Timeout:
            st.session_state.analysis_result = {
                'success': False,
                'error': "Request timed out. The server may be busy.",
                'details': {}
            }
            st.error("❌ Request timed out")
            
        except Exception as e:
            st.session_state.analysis_result = {
                'success': False,
                'error': f"Unexpected error: {str(e)}",
                'details': {}
            }
            st.error(f"❌ Error: {str(e)}")

def display_results():
    """Display the analysis results"""
    result = st.session_state.analysis_result
    
    if result['success']:
        st.markdown('<div class="result-section">', unsafe_allow_html=True)
        st.header("📊 Analysis Results")
        
        # Prediction
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Prediction", result['prediction'])
        
        with col2:
            if 'confidence' in result and result['confidence']:
                st.metric("Confidence", f"{result['confidence']:.1%}")
        
        # Additional details
        if result['details']:
            st.subheader("📋 Additional Information")
            for key, value in result['details'].items():
                st.write(f"**{key.title()}:** {value}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
    else:
        st.markdown('<div class="error-section">', unsafe_allow_html=True)
        st.header("❌ Analysis Failed")
        st.error(result['error'])
        
        if result['details']:
            st.subheader("Error Details")
            st.code(result['details'])
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main() 