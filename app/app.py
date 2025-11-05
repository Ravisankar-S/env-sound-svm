import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from src.utils import load_metrics, display_metrics_table, load_model
from src.predict_sound import predict_sound, get_best_kernel

st.set_page_config(page_title="Environmental Sound Classifier", layout="wide")

# ───────────────────────────────────────────────
st.title("🌦️ Environmental Sound Classification using SVM Kernels")
st.markdown(
    """
    This app uses **Support Vector Machines (SVMs)** to classify environmental sounds 
    like *rain, dog bark, sea waves, fire crackling,* and more.
    It compares different kernel types (Linear, Polynomial, RBF, Sigmoid) and finds which works best for this dataset.
    """
)

st.markdown(
    """
    <div style="padding:12px; border-radius:8px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); margin-bottom:16px;">
        <div style="color:#ffffff; font-weight:600; font-size:16px; margin-bottom:8px;">
            Sounds that can be classified:
        </div>
        <div style="display:flex; flex-wrap:wrap; gap:8px;">
            <span style="background-color:rgba(255,255,255,0.2); color:#ffffff; padding:6px 12px; border-radius:16px; font-size:13px; font-weight:500;">🐕 Dog</span>
            <span style="background-color:rgba(255,255,255,0.2); color:#ffffff; padding:6px 12px; border-radius:16px; font-size:13px; font-weight:500;">🌧️ Rain</span>
            <span style="background-color:rgba(255,255,255,0.2); color:#ffffff; padding:6px 12px; border-radius:16px; font-size:13px; font-weight:500;">🌊 Sea waves</span>
            <span style="background-color:rgba(255,255,255,0.2); color:#ffffff; padding:6px 12px; border-radius:16px; font-size:13px; font-weight:500;">👶 Crying baby</span>
            <span style="background-color:rgba(255,255,255,0.2); color:#ffffff; padding:6px 12px; border-radius:16px; font-size:13px; font-weight:500;">🕐 Ticking clock</span>
            <span style="background-color:rgba(255,255,255,0.2); color:#ffffff; padding:6px 12px; border-radius:16px; font-size:13px; font-weight:500;">🪚 Chainsaw</span>
            <span style="background-color:rgba(255,255,255,0.2); color:#ffffff; padding:6px 12px; border-radius:16px; font-size:13px; font-weight:500;">🔥 Crackling fire</span>
            <span style="background-color:rgba(255,255,255,0.2); color:#ffffff; padding:6px 12px; border-radius:16px; font-size:13px; font-weight:500;">🚁 Helicopter</span>
            <span style="background-color:rgba(255,255,255,0.2); color:#ffffff; padding:6px 12px; border-radius:16px; font-size:13px; font-weight:500;">🐓 Rooster</span>
            <span style="background-color:rgba(255,255,255,0.2); color:#ffffff; padding:6px 12px; border-radius:16px; font-size:13px; font-weight:500;">🤧 Sneezing</span>
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
    st.subheader("Upload a Sound File to Classify")
    st.caption(
        "Upload a short environmental sound clip (.wav/.ogg/.mp3). The system will automatically use the best-performing kernel for prediction."
    )

    uploaded = st.file_uploader("Choose a sound file", type=["wav", "ogg", "mp3"], key="sound_uploader")

    if uploaded:
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded.read())
        st.audio("temp_audio.wav")

        st.markdown("### 🧾 **RESULT:**")

        with st.spinner("🔍 Analyzing audio and predicting..."):
            try:
                label, conf = predict_sound("temp_audio.wav", kernel=best_kernel)
                conf_str = f"{conf:.3f}" if conf else "N/A"
            except Exception as e:
                st.error(f"Error predicting with {best_kernel}: {e}")
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

        st.markdown(
            f"""
            <div style="margin-top:12px; padding:10px; border-radius:8px; background-color:#e8f3ff; color:#000000;">
                <b>Best Kernel (based on metrics):</b>
                <span style="font-family: 'Georgia', serif; color:#0d47a1; font-weight:700; font-size:20px; margin-left:6px;">
                    {best_kernel.upper()}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        st.write("### 🔬 Kernel Comparison on This File")
        results = []
        for k in df_metrics["kernel"]:
            try:
                pred, conf = predict_sound("temp_audio.wav", kernel=k)
                conf_str = f"{conf:.3f}" if conf else "N/A"
                results.append({"Kernel": k.upper(), "Predicted": pred, "Confidence": conf_str})
            except Exception as e:
                results.append({"Kernel": k.upper(), "Predicted": f"Error: {e}", "Confidence": "—"})

        import pandas as pd
        results_df = pd.DataFrame(results)
        if not results_df.empty:
            results_df.index = pd.Index(range(1, len(results_df) + 1), name="S.No")
        st.table(results_df)

    else:
        st.info("📂 Upload a sound file to start classification.")

with tab2:
    st.subheader("📊 Kernel Performance Summary")
    st.caption(
        "Each kernel represents a different way of drawing decision boundaries in feature space. "
        "Higher accuracy and F1 scores indicate better generalization on unseen sounds."
    )
    st.markdown("<br>", unsafe_allow_html=True)
    display_metrics_table(df_metrics)
    
    st.markdown("---")
    st.subheader("💡 Understanding the Results")
    
    st.markdown("#### 🧠 Why RBF?")
    st.markdown(
        """
        - The **RBF (Radial Basis Function)** kernel is the default choice for SVMs because it's a powerful non-linear tool, perfect for complex audio data.
        - Unlike a linear kernel that draws a "straight fence," RBF can draw a "custom lasso" around complex, overlapping sound classes.
        - This flexibility allows it to find sophisticated patterns in the high-dimensional feature space.
        - Ideal for separating nuanced sounds like "rain" and "wind."
        """
    )
    
    st.markdown("#### 😮 The Surprising Strength of 'Linear'")
    st.markdown(
        """
        - The **linear kernel** performed surprisingly well (72.5% accuracy), which is a key finding that validates our feature extraction.
        - By transforming raw audio into features like MFCCs, we "un-tangled" the data so well that even a simple model could get good results.
        - The RBF kernel still wins by capturing the final, complex patterns that the linear model misses, giving it the slight edge.
        """
    )
    
    st.markdown("#### 📉 What About the Other Kernels?")
    st.markdown(
        """
        - The **Sigmoid kernel's** strong performance confirmed that a non-linear approach is correct for this data.
        - However, the **Poly kernel** was the clear loser, proving that just being non-linear isn't enough.
        - Its specific polynomial shape was the wrong fit for our sound features, demonstrating it was too rigid and less flexible than RBF for this specific task.
        """
    )

# ───────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "Made with ❤️ by <a href='https://www.linkedin.com/in/ravisankar-s-a3a881292/' target='_blank' style='text-decoration:none;'>Ravi</a>",
    unsafe_allow_html=True
)
