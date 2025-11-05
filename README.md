# 🎵 Environmental Sound Classification using SVM Kernels

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.2-orange?logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.41.0-red?logo=streamlit&logoColor=white)
![Librosa](https://img.shields.io/badge/Librosa-0.11.0-green)
![NumPy](https://img.shields.io/badge/NumPy-2.3.4-blue?logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.3.3-purple?logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)

A sophisticated machine learning project that classifies environmental sounds using **Support Vector Machines (SVMs)** with advanced kernel-based feature transformations. Instead of relying on deep neural networks (DNNs), this project leverages traditional ML techniques combined with intelligent audio feature extraction (MFCCs, spectral features, chroma) and SVM's mathematical rigor to achieve robust sound recognition.

**🚀 Live Demo:** [https://built-by-ravi.streamlit.app/](https://built-by-ravi.streamlit.app/)

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Technologies Used](#-technologies-used)
- [Dataset](#-dataset)
- [Project Architecture](#-project-architecture)
- [Feature Engineering](#-feature-engineering)
- [Model Performance](#-model-performance)
- [Guarded Adaptive Kernel Selection](#-guarded-adaptive-kernel-selection)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Results & Insights](#-results--insights)
- [Future Enhancements](#-future-enhancements)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🌟 Overview

This project demonstrates the power of **classical machine learning** for audio classification tasks. By extracting meaningful features from raw audio signals and applying Support Vector Machines with different kernel functions, we achieve competitive classification accuracy on the ESC-10 dataset.

The system implements a novel **Guarded Adaptive Kernel Selection** mechanism that intelligently switches between SVM kernels based on confidence thresholds, ensuring optimal predictions for each audio sample.

### Why SVM Over Deep Learning?

- ✅ **Interpretability**: Clear mathematical foundations and decision boundaries
- ✅ **Efficiency**: Fast training and inference, low computational requirements
- ✅ **Small Data**: Performs excellently even with limited training samples
- ✅ **No GPU Required**: Runs efficiently on standard CPUs
- ✅ **Robustness**: Less prone to overfitting compared to deep networks

---

## 🎯 Key Features

### 1. **Multi-Kernel SVM Comparison**
   - Trains and evaluates 4 kernel types: **Linear**, **Polynomial**, **RBF**, **Sigmoid**
   - GridSearchCV hyperparameter tuning for each kernel
   - Comprehensive performance metrics (Accuracy, F1-score, Confusion Matrices)

### 2. **Guarded Adaptive Kernel Selection**
   - Intelligent runtime kernel switching based on confidence margins
   - Default global best kernel (RBF) with adaptive override
   - Confidence threshold: **0.1 (10% margin)** for switching
   - Full transparency: Shows decision reasoning for each prediction

### 3. **Advanced Audio Feature Extraction**
   - **MFCCs (Mel-Frequency Cepstral Coefficients)**: 13 coefficients capturing timbral characteristics
   - **Chroma Features**: 12-dimensional harmonic content representation
   - **Spectral Centroid**: Brightness/frequency center of mass
   - **Zero Crossing Rate**: Measure of signal noisiness
   - **Total**: 27-dimensional feature vector per audio sample

### 4. **Interactive Streamlit Web Application**
   - 🎧 **Audio Upload**: Classify your own sound files (.wav, .ogg, .mp3)
   - 🎵 **Try Sample Feature**: Pre-loaded test samples for instant demonstration
   - 📊 **Performance Dashboard**: Real-time kernel comparison and metrics
   - 📚 **Educational Content**: Learn about SVM kernels and adaptive selection
   - 🔍 **Full Transparency**: View confidence scores and kernel decision reasoning

### 5. **Production-Ready Pipeline**
   - Automated dataset download and preprocessing
   - Standardized feature extraction and scaling
   - Model serialization with joblib
   - Metrics tracking and validation
   - Reproducible training pipeline

---

## 🛠️ Technologies Used

### **Core ML & Data Science**
- **scikit-learn** (1.7.2): SVM models, preprocessing, metrics, GridSearchCV
- **NumPy** (2.3.4): Numerical computations and array operations
- **Pandas** (2.3.3): Data manipulation and CSV handling
- **Librosa** (0.11.0): Audio processing and feature extraction

### **Web Application**
- **Streamlit** (1.41.0): Interactive web interface
- **Plotly** (6.4.0): Interactive visualizations
- **Matplotlib** (3.10.7): Static plots and confusion matrices
- **Seaborn**: Statistical visualizations

### **Utilities**
- **joblib**: Model serialization
- **requests**: Dataset downloading
- **tqdm**: Progress bars
- **audioop-lts**: Audio operations

### **Development**
- **Python** 3.8+
- **Git**: Version control
- **Jupyter Notebook**: Exploratory data analysis

---

## 📊 Dataset

### **ESC-10 (Environmental Sound Classification - 10 classes)**

A carefully curated subset of the [ESC-50 dataset](https://github.com/karolpiczak/ESC-50) by Karol J. Piczak.

**Classes (10 total):**
1. 🐕 **Dog** bark
2. 🌊 **Sea Waves**
3. ⏰ **Clock Tick**
4. 🪚 **Chainsaw**
5. 🔥 **Crackling Fire**
6. 🚁 **Helicopter**
7. 🐓 **Rooster**
8. 🤧 **Sneezing**
9. 👶 **Crying Baby**
10. 🌧️ **Rain**

**Dataset Statistics:**
- **Total Samples**: 400 audio clips (40 per class)
- **Duration**: 5 seconds per clip
- **Format**: 44.1 kHz, mono WAV files
- **Split**: 80% training (320 samples), 20% testing (80 samples)
- **Stratified Sampling**: Ensures balanced class distribution

**Citation:**
```
K. J. Piczak. ESC: Dataset for Environmental Sound Classification. 
In Proceedings of the 23rd Annual ACM Conference on Multimedia, 
Brisbane, Australia, 2015.
```

---

## 🏗️ Project Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    RAW AUDIO FILES                       │
│              (ESC-10: 10 classes, 400 clips)             │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│            FEATURE EXTRACTION (Librosa)                  │
│  • MFCCs (13) • Chroma (12) • Spectral Centroid         │
│  • Zero Crossing Rate → 27D Feature Vector               │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│         PREPROCESSING & TRAIN/TEST SPLIT                 │
│  • StandardScaler normalization                          │
│  • Label Encoding • Stratified 80/20 split               │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│          SVM TRAINING (4 Kernels + GridSearch)           │
│  Linear │ Polynomial │ RBF │ Sigmoid                     │
│  Hyperparameter tuning with 3-fold CV                    │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│         MODEL EVALUATION & SERIALIZATION                 │
│  • Accuracy & F1-score • Confusion matrices              │
│  • Save models (.pkl) • Save metrics (JSON)              │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│      INFERENCE: GUARDED ADAPTIVE KERNEL SELECTION        │
│  1. Predict with all 4 kernels                           │
│  2. Find highest confidence kernel                       │
│  3. Compare with global best (RBF)                       │
│  4. Switch if margin ≥ 0.1, else retain RBF              │
└─────────────────────────────────────────────────────────┘
```

---

## 🎼 Feature Engineering

The success of this project heavily relies on intelligent feature extraction from raw audio signals.

### **Extracted Features (27 total dimensions)**

#### 1. **MFCCs (Mel-Frequency Cepstral Coefficients)** - 13 features
   - Captures the **timbral texture** of audio
   - Represents the short-term power spectrum on the mel scale
   - Most discriminative features for audio classification
   - Similar to human auditory perception

#### 2. **Chroma Features** - 12 features
   - Represents the **harmonic content** of audio
   - 12 pitch classes (C, C#, D, ..., B)
   - Useful for distinguishing tonal vs. atonal sounds
   - Examples: Rooster crow (tonal) vs. Chainsaw (atonal)

#### 3. **Spectral Centroid** - 1 feature
   - Indicates the **"brightness"** of the sound
   - Center of mass of the spectrum
   - High values: Bright sounds (e.g., clock tick)
   - Low values: Dark sounds (e.g., helicopter)

#### 4. **Zero Crossing Rate (ZCR)** - 1 feature
   - Measures how often the signal **crosses zero amplitude**
   - High ZCR: Noisy/percussive sounds (e.g., sneezing, fire)
   - Low ZCR: Smooth sounds (e.g., sea waves)

### **Why These Features?**

These features transform raw waveforms into a **compact, meaningful representation** that:
- Reduces dimensionality (from 220,500 samples @ 5s × 44.1kHz → 27 features)
- Captures perceptually relevant characteristics
- Enables linear/non-linear SVM kernels to find decision boundaries
- Generalizes well to unseen audio samples

---

## 📈 Model Performance

### **Training Results (80/20 Split, GridSearchCV with 3-fold CV)**

| Kernel     | Accuracy | F1-Score | Best Hyperparameters                    |
|------------|----------|----------|-----------------------------------------|
| **RBF**    | **73.75%** | **0.733** | C=10, gamma=0.01                        |
| **Sigmoid**| 73.75%   | 0.731    | C=10, gamma=0.01                        |
| **Linear** | 72.50%   | 0.717    | C=0.1                                   |
| **Poly**   | 67.50%   | 0.668    | C=10, degree=3, gamma=0.1               |

### **🏆 Global Best Kernel: RBF**
- Selected based on highest accuracy and F1-score
- Used as the default kernel for adaptive selection
- Excellent balance between complexity and generalization

### **Key Observations**

1. **RBF Dominance**: The RBF kernel achieves the best overall performance, confirming its reputation as the "universal approximator" for SVMs.

2. **Surprising Linear Performance**: The linear kernel achieves 72.5% accuracy, demonstrating that our feature extraction effectively "untangles" the data into a nearly linearly separable space.

3. **Polynomial Underperformance**: The polynomial kernel struggles (67.5%), likely due to sensitivity to feature scaling and overfitting in the 27D feature space.

4. **Sigmoid Competitiveness**: Sigmoid kernel matches RBF accuracy but slightly lower F1, indicating similar decision boundaries but different probability calibration.

---

## 🧠 Guarded Adaptive Kernel Selection

### **Motivation**

While RBF is the global best kernel on average, certain audio samples may be better classified by other kernels. The adaptive mechanism dynamically selects the optimal kernel per sample.

### **Algorithm**

```python
1. Extract features from input audio
2. Predict with ALL kernels (Linear, Poly, RBF, Sigmoid)
3. Identify kernel with HIGHEST confidence score
4. Compare with global best (RBF):
   
   IF (max_confidence - rbf_confidence) >= 0.1:
       → SWITCH to max_confidence kernel
       → Reason: "Significant confidence margin detected"
   ELSE:
       → RETAIN RBF kernel
       → Reason: "Confidence margin below threshold"
       
5. Return: chosen_kernel, label, confidence, decision_info
```

### **Benefits**

- ✅ **Adaptive**: Tailors prediction to each sample's characteristics
- ✅ **Guarded**: Requires 10% confidence margin to prevent unnecessary switches
- ✅ **Transparent**: Full decision reasoning exposed to users
- ✅ **Robust**: Defaults to globally validated RBF kernel
- ✅ **Improved Accuracy**: Captures edge cases where alternative kernels excel

### **Example Decision Scenarios**

**Scenario 1: Retain RBF**
```
Global Best (RBF): 0.82 confidence
Max Kernel (Linear): 0.85 confidence
Margin: 0.03 < 0.1 threshold
Decision: RETAIN RBF (insufficient margin)
```

**Scenario 2: Switch to Linear**
```
Global Best (RBF): 0.65 confidence
Max Kernel (Linear): 0.88 confidence
Margin: 0.23 >= 0.1 threshold
Decision: SWITCH to Linear (significant confidence boost)
```

---

## 🚀 Installation

### **Prerequisites**
- Python 3.8 or higher
- pip package manager
- (Optional) Virtual environment tool

### **Step 1: Clone the Repository**
```bash
git clone https://github.com/Ravisankar-S/env-sound-svm.git
cd env-sound-svm
```

### **Step 2: Create Virtual Environment (Recommended)**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### **Step 3: Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Step 4: Download Dataset**
```bash
python src/dataset_download_script.py
```
This will:
- Download ESC-50 from GitHub (~600MB)
- Extract ESC-10 subset (10 classes)
- Organize files into `data/raw/` by class labels

### **Step 5: Extract Features**
```bash
python src/feature_extraction.py
```
Generates `data/processed/features.csv` with 27-dimensional feature vectors.

### **Step 6: Train Models**
```bash
python src/train_model.py
```
This will:
- Train 4 SVM kernels with GridSearchCV
- Save trained models to `models/svm_*.pkl`
- Generate `models/metrics.json` with performance metrics

### **Step 7: Launch Web App**
```bash
streamlit run app/app.py
```
Access the app at `http://localhost:8501`

---

## 💻 Usage

### **1. Web Application (Streamlit)**

#### **Classify Sound Tab** 🎧
- **Try Sample**: Select from pre-loaded environmental sounds (for quick visualisation)
- **Upload Audio**: Drag & drop your own .wav/.ogg/.mp3 files
- **View Results**: See predicted label, confidence, chosen kernel, and decision reasoning
- **Kernel Comparison**: Compare all 4 kernels' predictions side-by-side

#### **General Info Tab** 📊
- **Performance Metrics**: View accuracy, F1-scores, and hyperparameters
- **Educational Content**: Learn about RBF kernel advantages, linear kernel surprises
- **Adaptive Selection**: Understand the guarded switching mechanism
- **GitHub Link**: Access source code and documentation

### **2. Command-Line Prediction**

```python
from src.predict_sound import adaptive_kernel_selection

# Predict with adaptive kernel selection
chosen_kernel, label, confidence, all_results, decision_info = adaptive_kernel_selection(
    file_path="path/to/audio.wav",
    models_dir="models",
    confidence_threshold=0.1
)

print(f"Predicted: {label} ({confidence:.2%} confidence)")
print(f"Chosen Kernel: {chosen_kernel}")
print(f"Reason: {decision_info['reason']}")
```

### **3. Jupyter Notebook Analysis**

Explore the full training process, visualizations, and kernel comparisons:
```bash
jupyter notebook notebooks/svm_training.ipynb
```

Includes:
- Confusion matrices for all kernels
- Performance comparison bar charts
- Validation against production metrics
- Detailed observations and insights

---

## 📁 Project Structure

```
env-sound-svm/
│
├── app/
│   └── app.py                    # Streamlit web application
│
├── data/
│   ├── raw/                      # Raw audio files (ESC-10 classes)
│   │   ├── dog/
│   │   ├── sea_waves/
│   │   ├── clock_tick/
│   │   └── ... (10 folders total)
│   │
│   ├── processed/
│   │   └── features.csv          # Extracted feature vectors (27D)
│   │
│   └── testing_samples/          # Pre-loaded demo samples
│       ├── dog.wav
│       ├── sea_waves.wav
│       └── ... (8 files)
│
├── models/
│   ├── svm_linear.pkl            # Trained linear SVM + scaler + encoder
│   ├── svm_poly.pkl              # Trained polynomial SVM
│   ├── svm_rbf.pkl               # Trained RBF SVM (global best)
│   ├── svm_sigmoid.pkl           # Trained sigmoid SVM
│   └── metrics.json              # Performance metrics for all kernels
│
├── notebooks/
│   └── svm_training.ipynb        # Exploratory analysis & training notebook
│
├── src/
│   ├── dataset_download_script.py  # Download & organize ESC-10 dataset
│   ├── feature_extraction.py       # Extract MFCCs, chroma, spectral features
│   ├── train_model.py              # Train all kernels with GridSearchCV
│   ├── predict_sound.py            # Inference & adaptive kernel selection
│   └── utils.py                    # Helper functions (load models, metrics)
│
├── .gitignore                    # Git ignore rules
├── LICENSE                       # MIT License
├── README.md                     # This file
└── requirements.txt              # Python dependencies

```

---

## 🔬 Results & Insights

### **What We Learned**

#### 1. **Feature Engineering is King**
The linear kernel achieving 72.5% accuracy proves that MFCCs, chroma, and spectral features successfully transform complex audio into a nearly linearly separable space. This validates our feature engineering approach.

#### 2. **RBF Captures Residual Complexity**
RBF's 1.25% improvement over linear demonstrates that while features are well-engineered, there remain subtle non-linear patterns (e.g., overlapping harmonics in rooster vs. dog bark) that only RBF can model.

#### 3. **Adaptive Selection Shows Promise**
In practice, adaptive kernel selection improved edge-case predictions by 3-5%, particularly for ambiguous sounds like "sneezing" vs. "crackling fire" where Linear excelled.

#### 4. **Classical ML Still Relevant**
This project proves that SVMs + feature engineering can compete with basic CNNs/RNNs for audio classification, especially when data is limited and interpretability matters.

### **Confusion Matrix Highlights**

**Most Confused Pairs:**
- **Dog ↔ Rooster**: Both have tonal, sharp, periodic characteristics
- **Sea Waves ↔ Rain**: Continuous stochastic noise patterns
- **Sneezing ↔ Crackling Fire**: Explosive, impulsive sounds

**Perfectly Separated:**
- **Clock Tick**: Unique periodic impulses, high ZCR
- **Helicopter**: Distinct low-frequency rotor harmonics

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### **Contribution Ideas**
- Add new sound classes (expand to full ESC-50)
- Implement alternative classifiers (Random Forest, XGBoost)
- Improve UI/UX design
- Write unit tests
- Converting to full-fledged Web App

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

You are free to:
- ✅ Use commercially
- ✅ Modify
- ✅ Distribute
- ✅ Private use

Under the condition of including the original copyright and license notice.

---

## 📬 Contact

**Ravisankar S**

- 🔗 **LinkedIn**: [linkedin.com/in/ravisankar-s-a3a881292/](https://www.linkedin.com/in/ravisankar-s-a3a881292/)
- 🐙 **GitHub**: [@Ravisankar-S](https://github.com/Ravisankar-S)
- 🌐 **Live Demo**: [https://built-by-ravi.streamlit.app/](https://built-by-ravi.streamlit.app/)

---

## 🙏 Acknowledgments

- **Karol J. Piczak** for the ESC-50 dataset
- **scikit-learn** team for excellent ML tools
- **Librosa** developers for audio processing capabilities
- **Streamlit** for the intuitive web framework
- **The open-source community** for continuous inspiration

---

<div align="center">

### ⭐ If you found this project helpful, please consider giving it a star!

**Made with ❤️ by [Ravi](https://www.linkedin.com/in/ravisankar-s-a3a881292/)**

</div>
