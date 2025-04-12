import streamlit as st
import time
import logging
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import io

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Fire Detection System",
    page_icon="🔥",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load the YOLO model once and cache it"""
    return YOLO("fire_best.pt")

def process_image(image, model):
    """Process the image with the YOLO model and return results"""
    start_time = time.time()
    
    # Run the YOLO model prediction
    result = model.predict(image)
    
    # Extract bounding boxes and confidence scores
    xywh = result[0].boxes.xywh
    conf = result[0].boxes.conf
    
    # Format the results
    predictions = []
    for i in range(xywh.shape[0]):
        x, y, width, height = xywh[i]
        confidence = conf[i].item()
        predictions.append({
            "x": x.item(),
            "y": y.item(),
            "width": width.item(),
            "height": height.item(),
            "confidence": confidence
        })
    
    elapsed_time = time.time() - start_time
    
    return predictions, elapsed_time

def draw_bounding_boxes(image, predictions):
    """Draw bounding boxes on the image"""
    draw = ImageDraw.Draw(image)
    
    # Try to get a font, use default if unable to load
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()
    
    for pred in predictions:
        x, y = pred["x"], pred["y"]
        width, height = pred["width"], pred["height"]
        confidence = pred["confidence"]
        
        # Calculate box coordinates
        x1 = x - width/2
        y1 = y - height/2
        x2 = x + width/2
        y2 = y + height/2
        
        # Draw rectangle for the detection
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        
        # Draw label - ensure rectangle coordinates are valid
        label = f"Fire: {confidence:.2f}"
        text_width = len(label) * 7  # Approximate width of text
        
        # Place label above the box if there's space, otherwise below
        if y1 > 20:  # If there's space above the box
            label_y1 = max(0, y1 - 20)
            label_y2 = y1
            # Draw label background
            draw.rectangle([x1, label_y1, x1 + text_width, label_y2], fill="red")
            # Draw text
            draw.text((x1, label_y1), label, fill="white", font=font)
        else:
            # Place label below the box
            label_y1 = y2
            label_y2 = y2 + 20
            # Draw label background
            draw.rectangle([x1, label_y1, x1 + text_width, label_y2], fill="red")
            # Draw text
            draw.text((x1, label_y1), label, fill="white", font=font)
    
    return image

def main():
    # Load the model
    model = load_model()
    
    # App title and description
    st.title("🔥 Fire Detection System")
    st.markdown("""
    Upload an image to detect fires using a YOLO model.
    This application highlights potential fire hazards in images.
    """)
    
    # Create columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        # File uploader
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Display original image
            image = Image.open(uploaded_file)
            st.subheader("Original Image")
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Process button
            if st.button("Detect Fires"):
                with st.spinner("Processing image..."):
                    # Process the image and get predictions
                    predictions, elapsed_time = process_image(image, model)
                    
                    # Display processing information
                    st.success(f"Processing complete! Took {elapsed_time:.2f} seconds")
                    st.info(f"Found {len(predictions)} potential fire(s)")
                    
                    # Display result details in the second column
                    with col2:
                        st.subheader("Detection Results")
                        
                        if len(predictions) > 0:
                            # Draw bounding boxes on a copy of the image
                            result_image = image.copy()
                            result_image = draw_bounding_boxes(result_image, predictions)
                            
                            # Show the result image
                            st.image(result_image, caption="Detection Results", use_column_width=True)
                            
                            # Show prediction details
                            st.subheader("Detailed Results")
                            for i, pred in enumerate(predictions):
                                with st.expander(f"Fire #{i+1} (Confidence: {pred['confidence']:.2f})"):
                                    st.json(pred)
                        else:
                            st.info("No fires detected in the image")

if __name__ == "__main__":
    main()
