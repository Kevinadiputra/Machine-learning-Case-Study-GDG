"""
Lung Segmentation App with YOLOv8-seg
Streamlit Web Application for Medical Image Segmentation
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import sys
from pathlib import Path
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from inference import LungSegmentationModel
from utils import visualize_results, calculate_metrics

# Page config
st.set_page_config(
    page_title="Lung Segmentation App",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem;
        border-radius: 10px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize model
@st.cache_resource
def load_model():
    """Load YOLOv8 model"""
    try:
        model_path = Path(__file__).parent.parent / "models" / "best.pt"
        model = LungSegmentationModel(str(model_path))
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def main():
    # Header
    st.title("🫁 Lung Segmentation Application")
    st.markdown("---")
    st.markdown("""
    **Medical Image Segmentation using YOLOv8-seg**
    
    Upload an X-ray image to automatically segment:
    - Body
    - Cord (Spinal Cord)
    - Right Lung (Paru Kanan)
    - Left Lung (Paru Kiri)
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model info
        st.subheader("Model Information")
        st.info("""
        **Model:** YOLOv8n-seg
        **Input Size:** 640x640
        **Classes:** 4
        **mAP@0.5:** ~75%
        """)
        
        # Parameters
        st.subheader("Detection Parameters")
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="Minimum confidence score for detection"
        )
        
        iou_threshold = st.slider(
            "IoU Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Intersection over Union threshold for NMS"
        )
        
        show_labels = st.checkbox("Show Labels", value=True)
        show_confidence = st.checkbox("Show Confidence", value=True)
        
        st.markdown("---")
        
        # About
        st.subheader("ℹ️ About")
        st.markdown("""
        Developed for medical image analysis.
        
        **Technology:**
        - YOLOv8-seg
        - Streamlit
        - OpenCV
        """)
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("Failed to load model. Please check model file.")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # File uploader
    st.header("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an X-ray image...",
        type=["jpg", "jpeg", "png"],
        help="Upload a chest X-ray image for segmentation"
    )
    
    if uploaded_file is not None:
        # Read image
        image_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image_np = np.array(image)
        
        # Convert to RGB if needed
        if len(image_np.shape) == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        elif image_np.shape[2] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
        
        # Display original image
        st.header("🖼️ Original Image")
        st.image(image_np, caption="Input X-ray Image", use_column_width=True)
        
        # Progress bar
        with st.spinner("🔄 Processing image..."):
            start_time = time.time()
            # Run inference
            results = model.predict(
                image_np,
                conf_threshold=confidence_threshold,
                iou_threshold=iou_threshold
            )
            inference_time = time.time() - start_time
        
        st.success("✅ Segmentation complete!")
        
        # Display results
        st.header("🎯 Segmentation Results")
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs([
            "📊 Visualization",
            "📈 Metrics",
            "💾 Export"
        ])
        
        with tab1:
            st.subheader("Segmented Image")
            
            # Visualize results
            result_image = visualize_results(
                image_np,
                results,
                show_labels=show_labels,
                show_conf=show_confidence
            )
            
            st.image(result_image, caption="Segmentation Result", use_column_width=True)
            
            # Side-by-side comparison
            st.subheader("Comparison View")
            col1, col2 = st.columns(2)
            with col1:
                st.image(image_np, caption="Original", use_column_width=True)
            with col2:
                st.image(result_image, caption="Segmented", use_column_width=True)
        
        with tab2:
            st.subheader("Detection Metrics")
            
            # Calculate metrics
            metrics = calculate_metrics(results, inference_time)
            
            # Display metrics in cards
            cols = st.columns(4)
            
            with cols[0]:
                st.metric(
                    label="Total Detections",
                    value=metrics['total_detections']
                )
            
            with cols[1]:
                st.metric(
                    label="Average Confidence",
                    value=f"{metrics['avg_confidence']:.2%}"
                )
            
            with cols[2]:
                st.metric(
                    label="Processing Time",
                    value=f"{metrics['inference_time']:.3f}s"
                )
            
            with cols[3]:
                st.metric(
                    label="FPS",
                    value=f"{1/metrics['inference_time']:.1f}"
                )
            
            # Per-class metrics
            st.subheader("Per-Class Detection")
            if metrics['detections']:
                for detection in metrics['detections']:
                    with st.expander(f"🎯 {detection['class_name']} (Confidence: {detection['confidence']:.2%})"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Bounding Box:**")
                            st.write(f"- X: {detection['bbox'][0]:.1f}")
                            st.write(f"- Y: {detection['bbox'][1]:.1f}")
                            st.write(f"- Width: {detection['bbox'][2]:.1f}")
                            st.write(f"- Height: {detection['bbox'][3]:.1f}")
                        with col2:
                            st.write(f"**Confidence:** {detection['confidence']:.2%}")
            else:
                st.info("No detections found")
        
        with tab3:
            st.subheader("Export Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Download result image
                result_bytes = cv2.imencode('.png', cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))[1].tobytes()
                st.download_button(
                    label="📥 Download Result Image",
                    data=result_bytes,
                    file_name="lung_segmentation_result.png",
                    mime="image/png"
                )
            
            with col2:
                # Download metrics as JSON
                import json
                metrics_json = json.dumps(metrics, indent=2)
                st.download_button(
                    label="📥 Download Metrics (JSON)",
                    data=metrics_json,
                    file_name="segmentation_metrics.json",
                    mime="application/json"
                )

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Lung Segmentation Application v1.0 | YOLOv8-seg</p>
        <p>For research and educational purposes only</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
