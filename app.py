"""VisionGuard Streamlit Web Application.

Provides a clean, interactive user interface to upload or select images,
run deep learning anomaly detection, inspect reconstruction outputs,
and visualize explainable defect heatmaps.
"""

from pathlib import Path
import sys

# Ensure project root in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from PIL import Image
import streamlit as st
import torch
from torchvision import transforms

from src.detect import load_model_and_threshold
from src.explain import compute_anomaly_map, generate_heatmap_overlay, save_explanation
from src.model import compute_reconstruction_error, get_device

# Page Configuration
st.set_page_config(
    page_title="VisionGuard — Anomaly Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #e9ecef;
    }
    .badge-normal {
        background-color: #d4edda;
        color: #155724;
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 1.2rem;
        display: inline-block;
    }
    .badge-anomaly {
        background-color: #f8d7da;
        color: #721c24;
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 1.2rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_cached_model(checkpoint_path: str = "checkpoints/visionguard_model.pth"):
    """Cache the loaded model in memory for fast interactive inference."""
    device = get_device()
    model, threshold, img_size, category = load_model_and_threshold(checkpoint_path, device=device)
    return model, threshold, img_size, category, device


def main():
    st.title("🔍 VisionGuard")
    st.subheader("Explainable Deep Learning Image Anomaly Detection")
    st.write(
        "VisionGuard trains a Convolutional Autoencoder strictly on normal industrial images. "
        "During inference, anomalous patterns fail to reconstruct accurately, producing high "
        "reconstruction errors and localized anomaly heatmaps."
    )

    # Checkpoint check
    ckpt_path = Path("checkpoints/visionguard_model.pth")
    if not ckpt_path.exists():
        st.error(
            "⚠️ Model checkpoint not found at `checkpoints/visionguard_model.pth`. "
            "Please train the model first by running `python src/train.py`."
        )
        return

    try:
        model, default_threshold, img_size, category, device = get_cached_model(str(ckpt_path))
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    # Sidebar
    st.sidebar.header("⚙️ Model Configuration")
    st.sidebar.markdown(f"**Dataset Category:** `{category}`")
    st.sidebar.markdown(f"**Inference Device:** `{device.type.upper()}`")
    st.sidebar.markdown(f"**Trained Resolution:** `{img_size}x{img_size}`")
    st.sidebar.markdown(f"**Default Threshold:** `{default_threshold:.6f}`")

    # Threshold override slider
    threshold_slider = st.sidebar.slider(
        "Decision Threshold",
        min_value=float(default_threshold * 0.5),
        max_value=float(default_threshold * 2.0),
        value=float(default_threshold),
        step=float(default_threshold * 0.05),
        format="%.6f",
        help="Images with reconstruction MSE score above this value are classified as ANOMALOUS.",
    )

    # Sample images selector
    st.sidebar.header("📁 Sample Images")
    sample_options = {
        "Custom Upload": None,
        "Normal Sample (Good)": "data/bottle/test/good/042.png",
        "Defect: Broken Small": "data/bottle/test/broken_small/000.png",
        "Defect: Broken Large": "data/bottle/test/broken_large/022.png",
        "Defect: Contamination": "data/bottle/test/contamination/062.png",
    }
    selected_sample = st.sidebar.selectbox("Choose a test sample:", list(sample_options.keys()))

    # Input image handling
    input_image = None
    image_name = "Uploaded Image"

    if selected_sample != "Custom Upload" and sample_options[selected_sample]:
        sample_path = Path(sample_options[selected_sample])
        if sample_path.exists():
            input_image = Image.open(sample_path).convert("RGB")
            image_name = sample_path.name
        else:
            st.sidebar.warning(f"Sample not found: {sample_path}")

    uploaded_file = st.file_uploader(
        "Or upload your own image (PNG, JPG, BMP):",
        type=["png", "jpg", "jpeg", "bmp"],
    )
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file).convert("RGB")
        image_name = uploaded_file.name

    if input_image is None:
        st.info("👆 Select a sample image from the sidebar or upload an image above to begin.")
        return

    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    tensor_input = transform(input_image).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        recon_tensor = model(tensor_input)
        error = compute_reconstruction_error(tensor_input, recon_tensor, reduction="none")
        score = float(error.item())

    is_anomalous = score >= threshold_slider
    prediction = "ANOMALOUS" if is_anomalous else "NORMAL"

    # NumPy arrays for visualization
    orig_np = tensor_input.squeeze(0).permute(1, 2, 0).cpu().numpy()
    recon_np = recon_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # Generate Anomaly Map and Heatmap Overlay
    anomaly_map = compute_anomaly_map(orig_np, recon_np)
    overlay = generate_heatmap_overlay(orig_np, anomaly_map)

    # Prediction Banner
    st.markdown("---")
    if is_anomalous:
        st.markdown(
            f"<div class='badge-anomaly'>🚨 ANOMALOUS — Defect Detected</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='badge-normal'>✅ NORMAL — Inspection Passed</div>",
            unsafe_allow_html=True,
        )

    st.write("")

    # Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Anomaly Score (MSE)", f"{score:.6f}")
    m2.metric("Decision Threshold", f"{threshold_slider:.6f}")
    m3.metric("Prediction", prediction)
    m4.metric("Status", "Defective" if is_anomalous else "Pass")

    # Image Visualization Columns
    st.markdown("### 🖼️ Explainable Visual Inspection")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.image(np.clip(orig_np, 0.0, 1.0), caption="1. Original Input Image", use_container_width=True)
    with col2:
        st.image(np.clip(recon_np, 0.0, 1.0), caption="2. Autoencoder Reconstruction", use_container_width=True)
    with col3:
        st.image(anomaly_map, caption="3. Error Heatmap", clamp=True, use_container_width=True)
    with col4:
        st.image(overlay, caption="4. Heatmap Overlay on Original", use_container_width=True)

    # Export visualization button
    output_preview_path = Path("outputs") / f"streamlit_{image_name}"
    save_explanation(
        original=orig_np,
        reconstructed=recon_np,
        anomaly_map=anomaly_map,
        overlay=overlay,
        save_path=output_preview_path,
        score=score,
        threshold=threshold_slider,
        prediction=prediction,
        sample_name=image_name,
    )

    if output_preview_path.exists():
        with open(output_preview_path, "rb") as f:
            st.download_button(
                label="📥 Download Full Explanation Report",
                data=f.read(),
                file_name=f"visionguard_{image_name}",
                mime="image/png",
            )


if __name__ == "__main__":
    main()
