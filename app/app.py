import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from src.utils import load_metrics, display_metrics_table, load_model
from src.predict_sound import predict_sound, get_best_kernel, adaptive_kernel_selection

st.set_page_config(page_title="Environmental Sound Classifier", layout="wide")

# ───────────────────────────────────────────────
st.title("Environmental Sound Classification using SVM Kernels")
st.markdown(
    """
    This app uses **Support Vector Machines (SVMs)** to classify environmental sounds 
    like *rain, dog bark, sea waves, fire crackling,* and more.
    It compares different kernel types (Linear, Polynomial, RBF, Sigmoid) and finds which works best for this dataset.

    - 🎧 <a href="#" style="text-decoration:none; color:#0066cc; font-weight:500;"> **Classify Sound Tab:** </a>Upload audio samples and receive instant predictions with adaptive kernel selection.
    - 📊 <a href="#" style="text-decoration:none; color:#0066cc; font-weight:500;"> **General Info Tab:** </a> Explore kernel performance metrics, understand adaptive selection, and learn SVM-Kernel concepts.
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="padding:12px; border-radius:8px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); margin-bottom:16px;">
        <div style="color:#ffffff; font-weight:600; font-size:16px; margin-bottom:8px;">
            Sounds that can be classified:
        </div>
        <div style="display:flex; flex-wrap:wrap; gap:8px;">
            <span style="background-color:rgba(255,255,255,0.2); color:#ffffff; padding:6px 12px; border-radius:16px; font-size:13px; font-weight:500;">Dog</span>
            <span style="background-color:rgba(255,255,255,0.2); color:#ffffff; padding:6px 12px; border-radius:16px; font-size:13px; font-weight:500;">Rain</span>
            <span style="background-color:rgba(255,255,255,0.2); color:#ffffff; padding:6px 12px; border-radius:16px; font-size:13px; font-weight:500;">Sea waves</span>
            <span style="background-color:rgba(255,255,255,0.2); color:#ffffff; padding:6px 12px; border-radius:16px; font-size:13px; font-weight:500;">Crying baby</span>
            <span style="background-color:rgba(255,255,255,0.2); color:#ffffff; padding:6px 12px; border-radius:16px; font-size:13px; font-weight:500;">Ticking clock</span>
            <span style="background-color:rgba(255,255,255,0.2); color:#ffffff; padding:6px 12px; border-radius:16px; font-size:13px; font-weight:500;">Chainsaw</span>
            <span style="background-color:rgba(255,255,255,0.2); color:#ffffff; padding:6px 12px; border-radius:16px; font-size:13px; font-weight:500;">Crackling fire</span>
            <span style="background-color:rgba(255,255,255,0.2); color:#ffffff; padding:6px 12px; border-radius:16px; font-size:13px; font-weight:500;">Helicopter</span>
            <span style="background-color:rgba(255,255,255,0.2); color:#ffffff; padding:6px 12px; border-radius:16px; font-size:13px; font-weight:500;">Rooster</span>
            <span style="background-color:rgba(255,255,255,0.2); color:#ffffff; padding:6px 12px; border-radius:16px; font-size:13px; font-weight:500;">Sneezing</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")
# ───────────────────────────────────────────────

try:
    df_metrics = load_metrics("models")
except Exception as e:
    st.error(f"Error loading metrics.json: {e}")
    st.stop()

best_kernel = get_best_kernel("models")

tab1, tab2 = st.tabs(["🎧 Classify Sound", "📊 General Info"])

with tab1:
    # Try Sample Feature
    st.markdown("#### 🎵 Try a Sample Audio")
    col1, col2 = st.columns([3, 1])
    
    sample_files = {
        "-- Select a sample --": None,
        "🐕 Dog": "data/testing_samples/dog.wav",
        "🌊 Sea Waves": "data/testing_samples/sea_waves.wav",
        "⏰ Clock Tick": "data/testing_samples/clock_tick.wav",
        "🪚 Chainsaw": "data/testing_samples/chainsaw.wav",
        "🔥 Crackling Fire": "data/testing_samples/crackling_fire.wav",
        "🚁 Helicopter": "data/testing_samples/helicopter.wav",
        "🐓 Rooster": "data/testing_samples/rooster.wav",
        "🤧 Sneezing": "data/testing_samples/sneezing.wav"
    }
    
    with col1:
        selected_sample = st.selectbox(
            "Choose a pre-loaded sample to test the model:",
            options=list(sample_files.keys()),
            key="sample_selector"
        )
    
    st.markdown("#### 📤 Or Upload Your Own Audio")
    uploaded = st.file_uploader("Choose a sound file", type=["wav", "ogg", "mp3"], key="sound_uploader")

    # Determine which audio source to use
    audio_file = None
    audio_source = None
    
    if selected_sample and selected_sample != "-- Select a sample --":
        audio_file = sample_files[selected_sample]
        audio_source = "sample"
    elif uploaded:
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded.read())
        audio_file = "temp_audio.wav"
        audio_source = "uploaded"

    if audio_file:
        st.audio(audio_file)
        
        if audio_source == "sample":
            st.info(f"🎵 **Using sample:** {selected_sample}")

        st.markdown("### 🧾 **RESULT:**")

        with st.spinner("🔍 Analyzing audio with adaptive kernel selection..."):
            try:
                chosen_kernel, label, conf, all_results, decision_info = adaptive_kernel_selection(
                    audio_file, 
                    models_dir="models", 
                    confidence_threshold=0.1
                )
                conf_str = f"{conf:.3f}" if conf else "N/A"
            except Exception as e:
                st.error(f"Error during adaptive prediction: {e}")
                st.stop()

        st.markdown(
            f"""
            <div style="margin-top:8px; padding:12px; border-radius:8px; background-color:#fff9f6;">
                <div style="font-size:18px; color:#222;">
                    🎯 <span style="font-weight:600;">Predicted Label:</span>
                    <span style="font-family: 'Georgia', serif; color: #d43f3a; font-size:28px; font-weight:700;">{label}</span>
                </div>
                <div style="color:#666666; font-size:16px; margin-top:6px;">
                    <span style="font-weight:400;">Confidence:</span>
                    <span style="font-family: 'Georgia', serif; color:#333; font-weight:700; font-size:18px;">{conf_str}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if decision_info["switched"]:
            kernel_color = "#28a745"
            icon = "🔄"
        else:
            kernel_color = "#0d47a1"
            icon = "✓"
        
        st.markdown(
            f"""
            <div style="margin-top:12px; padding:10px; border-radius:8px; background-color:#e8f3ff; color:#000000;">
                <div style="margin-bottom:6px;">
                    <b>Selected Kernel (Adaptive):</b>
                    <span style="font-family: 'Georgia', serif; color:{kernel_color}; font-weight:700; font-size:20px; margin-left:6px;">
                        {icon} {chosen_kernel.upper()}
                    </span>
                </div>
                <div style="font-size:13px; color:#555; margin-top:4px;">
                    {decision_info["reason"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        st.write("### 🔬 Kernel Comparison on This File")
        
        import pandas as pd
        results_data = []
        for r in all_results:
            conf_val = r["confidence"]
            conf_display = f"{conf_val:.3f}" if isinstance(conf_val, (int, float)) else "N/A"
            
            kernel_name = r["kernel"].upper()
            if r["kernel"] == chosen_kernel:
                kernel_name += " ⭐"
            
            results_data.append({
                "Kernel": kernel_name,
                "Predicted": r["label"],
                "Confidence": conf_display
            })
        
        results_df = pd.DataFrame(results_data)
        if not results_df.empty:
            results_df.index = pd.Index(range(1, len(results_df) + 1), name="S.No")
        st.table(results_df)
        
        st.markdown(
            f"""
            <div style="padding:12px; background-color:#d1ecf1; border-left:4px solid #0c5460; border-radius:4px;">
                <span style="color:#0c5460;">
                    ℹ️ <b>Adaptive Selection Info:</b> Global best kernel is <b>{decision_info['global_best_kernel'].upper()}</b> 
                    (confidence: {decision_info['global_best_confidence']:.3f}). 
                    Highest confidence kernel is <b>{decision_info['max_confidence_kernel'].upper()}</b> 
                    (confidence: {decision_info['max_confidence']:.3f}). 
                    Confidence difference: <b>{decision_info['confidence_diff']:.3f}</b>. 
                    Threshold for switching: <b>{decision_info['threshold']}</b>.
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown(
            """
            <div style="margin-top:8px; font-size:14px;">
                💡 <a href="https://built-by-ravi.streamlit.app/#guarded-adaptive-kernel-selection" style="text-decoration:none; color:#0066cc; font-weight:500;">
                Know more about Adaptive Kernel Selection →
                </a>
                <span style="color:#666; font-size:12px;">(See General Info tab)</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        st.info("📂 Upload a sound file to start classification.")

with tab2:
    st.subheader("📊 Kernel Performance Snapshot")
    st.caption(
        """
         - This model is trained on a 10-class subset called ESC-10 of the  <a href='https://github.com/karolpiczak/ESC-50' target='_blank' style='text-decoration:none;'>**karolpiczak/ESC-50**</a> dataset. 
         - Each kernel represents a different way of drawing decision boundaries in feature space. 
         - Higher accuracy and F1 scores indicate better generalization on unseen sounds. 
         - For detailed understanding, refer to the <a href='https://github.com/Ravisankar-S/env-sound-svm/blob/main/notebooks/svm_training.ipynb' target='_blank' style='text-decoration:none;'>**svm_training.ipynb**</a> file for graphs and analysis.
        """,
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)
    display_metrics_table(df_metrics)
    
    st.markdown("---")
    st.subheader("💡 Understanding the Results")
    
    st.markdown("#### Why RBF?")
    st.markdown(
        """
        - The **RBF (Radial Basis Function)** kernel is the default choice for SVMs because it's a powerful non-linear tool, perfect for complex audio data.
        - Unlike a linear kernel that draws a "straight fence," RBF can draw a "custom lasso" around complex, overlapping sound classes.
        - This flexibility allows it to find sophisticated patterns in the high-dimensional feature space.
        - Ideal for separating nuanced sounds like "rain" and "wind."
        """
    )
    
    st.markdown("#### The Surprising Strength of 'Linear'")
    st.markdown(
        """
        - The **linear kernel** performed surprisingly well (72.5% accuracy), which is a key finding that validates our feature extraction.
        - By transforming raw audio into features like MFCCs, we "un-tangled" the data so well that even a simple model could get good results.
        - The RBF kernel still wins by capturing the final, complex patterns that the linear model misses, giving it the slight edge.
        """
    )
    
    st.markdown("#### What About the Other Kernels?")
    st.markdown(
        """
        - The **Sigmoid kernel's** strong performance confirmed that a non-linear approach is correct for this data.
        - However, the **Poly kernel** was the clear loser, proving that just being non-linear isn't enough.
        - Its specific polynomial shape was the wrong fit for our sound features, demonstrating it was too rigid and less flexible than RBF for this specific task.
        """
    )
    
    st.markdown("---")
    st.markdown('<div id="guarded-adaptive-kernel-selection"></div>', unsafe_allow_html=True)
    st.subheader("🔄 Guarded Adaptive Kernel Selection")
    st.markdown("#### How It Works")
    st.markdown(
        """
        - **Step 1:** Predict with all kernels and compute confidence scores for each.
        - **Step 2:** Identify the kernel with the highest confidence for this specific sample.
        - **Step 3:** Compare against the global best kernel (RBF) confidence.
        - **Step 4:** If confidence difference ≥ 0.1, switch to the higher-confidence kernel; otherwise, retain RBF.
        """
    )
    
    st.markdown("#### Why Adaptive Selection?")
    st.markdown(
        """
        - **Local Adaptability:** Different kernels may perform better on specific sound samples due to their unique characteristics.
        - **Sample-Specific Optimization:** While RBF is globally optimal, individual samples may benefit from other kernels.
        - **Confidence-Guided:** Only switches when there's a significant confidence advantage (≥10% improvement).
        - **Guarded Approach:** Prevents unnecessary switching for marginal gains, maintaining stability.
        """
    )

# ───────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "Made with ❤️ by <a href='https://www.linkedin.com/in/ravisankar-s-a3a881292/' target='_blank' style='text-decoration:none;'>Ravi</a>",
    unsafe_allow_html=True
)
