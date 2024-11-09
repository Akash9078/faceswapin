import streamlit as st
import requests
from PIL import Image
import io
import tempfile
import os
from urllib.parse import urlparse

def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def download_image_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))
    except Exception as e:
        st.error(f"Error downloading image from URL: {str(e)}")
        return None

def main():
    st.title("Face Swapper Application 👤")
    st.write("Upload source and target images or provide URLs to swap faces!")

    # Create two columns for source and target inputs
    col1, col2 = st.columns(2)

    with col1:
        st.header("Source Image")
        source_option = st.radio("Choose source input:", ["Upload File", "URL"], key="source")
        
        if source_option == "Upload File":
            source_image = st.file_uploader("Upload source face image", type=['jpg', 'jpeg', 'png'])
        else:
            source_url = st.text_input("Enter source image URL")
            if source_url and is_valid_url(source_url):
                source_image = download_image_from_url(source_url)
            else:
                source_image = None

        if source_image:
            st.image(source_image, caption="Source Image", use_column_width=True)

    with col2:
        st.header("Target Image")
        target_option = st.radio("Choose target input:", ["Upload File", "URL"], key="target")
        
        if target_option == "Upload File":
            target_image = st.file_uploader("Upload target image", type=['jpg', 'jpeg', 'png'])
        else:
            target_url = st.text_input("Enter target image URL")
            if target_url and is_valid_url(target_url):
                target_image = download_image_from_url(target_url)
            else:
                target_image = None

        if target_image:
            st.image(target_image, caption="Target Image", use_column_width=True)

    # Advanced options in an expander
    with st.expander("Advanced Options"):
        col3, col4 = st.columns(2)
        
        with col3:
            background_enhance = st.checkbox("Background Enhancement", True)
            face_upsample = st.checkbox("Face Upsample", True)
            
        with col4:
            upscale = st.slider("Upscale Factor", 1, 4, 2)
            codeformer_fidelity = st.slider("CodeFormer Fidelity", 0.0, 1.0, 0.7)

    # Processing options
    processing_option = st.radio(
        "Choose Processing Option",
        ["Basic Face Swap", "Face Swap with Restoration"]
    )

    if st.button("Process Images"):
        if source_image is None or target_image is None:
            st.error("Please provide both source and target images!")
            return

        try:
            # Show processing message
            with st.spinner("Processing images... Please wait."):
                # Save images to temporary files
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_source:
                    if source_option == "Upload File":
                        tmp_source.write(source_image.getvalue())
                    else:
                        source_image.save(tmp_source, format='PNG')
                    source_path = tmp_source.name

                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_target:
                    if target_option == "Upload File":
                        tmp_target.write(target_image.getvalue())
                    else:
                        target_image.save(tmp_target, format='PNG')
                    target_path = tmp_target.name

                # Prepare API endpoint and data based on processing option
                if processing_option == "Basic Face Swap":
                    endpoint = "http://localhost:8000/swap-face/"
                    data = {
                        "source_url": source_url if source_option == "URL" else f"file://{source_path}",
                        "target_url": target_url if target_option == "URL" else f"file://{target_path}"
                    }
                else:
                    endpoint = "http://localhost:8000/swap-face-restore/"
                    data = {
                        "source_url": source_url if source_option == "URL" else f"file://{source_path}",
                        "target_url": target_url if target_option == "URL" else f"file://{target_path}",
                        "background_enhance": background_enhance,
                        "face_upsample": face_upsample,
                        "upscale": upscale,
                        "codeformer_fidelity": codeformer_fidelity
                    }

                # Make API request
                response = requests.post(endpoint, json=data)
                
                if response.status_code == 200:
                    # Convert response content to image
                    result_image = Image.open(io.BytesIO(response.content))
                    
                    # Display result
                    st.success("Face swap completed successfully!")
                    st.image(result_image, caption="Result Image", use_column_width=True)
                    
                    # Add download button
                    buf = io.BytesIO()
                    result_image.save(buf, format="PNG")
                    st.download_button(
                        label="Download Result",
                        data=buf.getvalue(),
                        file_name="face_swap_result.png",
                        mime="image/png"
                    )
                else:
                    st.error(f"Error: {response.text}")

                # Cleanup temporary files
                os.unlink(source_path)
                os.unlink(target_path)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()