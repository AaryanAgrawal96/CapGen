import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# ---------------------------
# Model setup
# ---------------------------
@st.cache_resource
def load_caption_model():
    """Load the image captioning processor and model."""
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

def generate_caption(image, processor, model):
    """Generate a caption for the uploaded image."""
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# ---------------------------
# Streamlit App
# ---------------------------
def main():
    st.set_page_config(
        page_title="CapGen",
        page_icon="🖼️",
        layout="centered",
        initial_sidebar_state="expanded"
    )

    # Title and description
    st.title("CapGen 🖼️")
    st.markdown(
        """
        **Generate smart captions for any image in seconds!**  
        Simply upload an image, and CapGen will provide a meaningful, descriptive caption.
        """
    )

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        # Resize image to small width for display
        max_width = 200
        aspect_ratio = image.height / image.width
        new_height = int(max_width * aspect_ratio)
        image_small = image.resize((max_width, new_height))
        st.image(image_small, caption='Uploaded Image', use_container_width=False)

        # Load model
        with st.spinner("Loading captioning model..."):
            processor, model = load_caption_model()

        # Generate caption
        with st.spinner("Generating caption..."):
            caption = generate_caption(image, processor, model)

        # Display caption with improved styling and copy button
        st.success("Caption generated!")
        st.markdown("### 🧾 Caption:")
        st.text_area(
            label="",
            value=caption,
            height=100,
            max_chars=None,
            key=f"caption_area_{uploaded_file.name if uploaded_file else ''}"
        )

    # Sidebar with info
    with st.sidebar:
        st.header("About CapGen")
        st.write(
            """
            - Uses state-of-the-art image captioning model  
            - Fully local processing, no external API required  
            - Fast and easy to use for any image
            """
        )
        st.info("Tip: Upload clear images for better captions!")

if __name__ == "__main__":
    main()
