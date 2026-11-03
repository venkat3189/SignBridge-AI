# SignBridge-AI
### Breaking Communication Barriers Through Artificial Intelligence

<p align="center">

**AI-Powered Indian Sign Language Recognition & Communication Assistance**

</p>

---

## 1. Project Overview

SignBridge-AI is an Artificial Intelligence and Machine Learning-based project designed to bridge the communication gap between individuals who use sign language and people who may not understand it.

The system uses computer vision, hand-landmark extraction, body-pose analysis and machine learning to recognise isolated sign-language gestures and convert them into understandable text.

By combining multiple machine learning architectures with a multimodal feature-extraction pipeline, SignBridge-AI aims to make sign-language recognition more accessible, efficient and practical.

The project currently explores three model architectures:

- Random Forest
- Support Vector Machine (SVM)
- Gated Recurrent Unit (GRU)

The system uses MediaPipe-based landmark extraction to capture hand movements and upper-body pose information, providing meaningful numerical features for sign recognition.

**Our vision:** Make communication more inclusive through intelligent, accessible and technology-driven solutions.

---

## 2. Problem Statement

Communication is an essential part of everyday life. However, individuals who communicate through sign language can face difficulties interacting with people who do not understand it.

Traditional communication methods may require a human interpreter or someone familiar with the same sign language.

This creates challenges in educational institutions, workplaces, healthcare environments and everyday social interactions.

Although sign-language recognition technologies exist, developing a reliable recognition system requires handling variations in hand positions, body movements, gesture duration and recording conditions.

### Our Proposed Solution

SignBridge-AI addresses this challenge by developing an AI-powered sign recognition system that:

1. Captures visual information from sign-language videos or a webcam.
2. Extracts meaningful hand and body-pose landmarks.
3. Processes the extracted features using machine learning models.
4. Predicts the corresponding sign from the trained vocabulary.
5. Converts recognised signs into readable text.
6. Provides a foundation for sentence construction and speech-based communication.

---

## 3. Project Objectives

- Develop an AI-based isolated sign-language recognition system.
- Extract hand and body-pose landmarks using computer vision.
- Investigate multimodal feature extraction for improved recognition.
- Train and evaluate multiple machine learning architectures.
- Compare model performance using standard evaluation metrics.
- Develop an accessible interface for sign-language communication.
- Explore speech output and sentence construction capabilities.
- Create a foundation for future real-time sign-language applications.

---

## 4. Key Features

### Current Machine Learning Capabilities

| Feature | Description |
|---|---|
| Hand Landmark Extraction | Extracts hand keypoints using MediaPipe |
| Body Pose Analysis | Uses upper-body pose information |
| Feature Engineering | Converts visual landmarks into numerical features |
| Multimodal Learning | Combines hand and body-pose information |
| Multiple ML Models | Implements Random Forest, SVM and GRU |
| Model Evaluation | Measures accuracy, precision, recall and F1-score |
| Model Persistence | Saves trained models for subsequent inference |

### Planned Application Capabilities

- Live webcam-based sign recognition.
- Real-time prediction visualisation.
- Sign-to-text translation.
- Sentence construction.
- Text-to-speech communication.
- Translation history.
- Model performance analytics.

*Application features should be marked as completed only after their implementation and testing.*

---

## 5. System Architecture

SignBridge-AI follows a modular machine learning architecture.

```text
             INPUT VIDEO / WEBCAM
                      |
                      v
             VIDEO PROCESSING
                      |
                      v
          MEDIAPIPE LANDMARK EXTRACTION
                      |
             +--------+--------+
             |                 |
             v                 v
        HAND LANDMARKS     BODY POSE
             |                 |
             +--------+--------+
                      |
                      v
             FEATURE EXTRACTION
                      |
                      v
             DATA PREPROCESSING
                      |
                      v
              MODEL TRAINING
                      |
          +-----------+-----------+
          |           |           |
          v           v           v
     RANDOM       SVM MODEL    GRU MODEL
     FOREST
          |           |           |
          +-----------+-----------+
                      |
                      v
               SIGN PREDICTION
                      |
                      v
                TEXT OUTPUT
                      |
                      v
           COMMUNICATION INTERFACE
```

### Architecture Explanation

**1. Input Acquisition**

The system processes sign-language video samples. Webcam input is intended for the live application.

**2. Landmark Extraction**

MediaPipe identifies important hand and body-pose landmarks from video frames.

**3. Feature Extraction**

The system converts the extracted landmarks into numerical representations.

The current feature configuration uses 182 features per frame, combining hand information and upper-body pose information.

**4. Data Preprocessing**

The extracted features are normalised, organised and prepared for model training.

**5. Model Training**

Multiple machine learning architectures are trained and evaluated to investigate their suitability for sign recognition.

**6. Prediction**

The trained models classify input features into one of the supported sign categories.

**7. Communication Output**

The predicted sign can be presented as text, with sentence construction and speech output planned as additional communication capabilities.

---

## 6. Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Computer Vision | OpenCV |
| Landmark Detection | MediaPipe |
| Numerical Computing | NumPy |
| Data Processing | Pandas |
| Machine Learning | Scikit-learn |
| Deep Learning | TensorFlow, Keras |
| Model Architectures | Random Forest, SVM, GRU |
| Model Persistence | Joblib |
| Application Framework | Streamlit |
| Data Visualisation | Plotly |
| Database | SQLite (planned application integration) |
| Speech Processing | Pyttsx3, Faster Whisper (planned integration) |
| Development Environment | Visual Studio Code |

---

## 7. Dataset Information

The current training pipeline uses an isolated sign-language video dataset organised into 30 sign categories.

### Dataset Statistics

| Parameter | Value |
|---|---:|
| Total processed videos | 539 |
| Sign categories | 30 |
| Training samples | 298 |
| Validation samples | 134 |
| Testing samples | 107 |

The dataset was divided into training, validation and testing subsets using recording-run separation.

**Important:** This split is not claimed to be signer-independent. The evaluation results should therefore be interpreted within the scope of this dataset and splitting strategy.

### Data Processing Workflow

1. Load the sign-language videos.
2. Detect hands and body landmarks.
3. Extract frame-level features.
4. Identify relevant gesture segments.
5. Normalise and resample the extracted sequences.
6. Save the processed landmark data.
7. Prepare training, validation and testing datasets.

---

## 8. Machine Learning Models

SignBridge-AI investigates three different machine learning architectures.

### 8.1 Random Forest

Random Forest is an ensemble learning algorithm that combines multiple decision trees to generate predictions.

It is used as a classical machine learning baseline for sign classification.

**Advantages:**
- Handles numerical features effectively.
- Captures nonlinear relationships.
- Provides a useful baseline for comparison.

### 8.2 Support Vector Machine (SVM)

Support Vector Machine is a supervised learning algorithm that identifies decision boundaries between different classes.

It provides another classical machine learning baseline for evaluating the extracted landmark features.

**Advantages:**
- Effective for structured feature representations.
- Can model complex decision boundaries using suitable kernels.
- Useful for comparing classification approaches.

### 8.3 Gated Recurrent Unit (GRU)

GRU is a recurrent neural network architecture designed to process sequential data.

Since sign language involves movements over time, GRU provides a way to investigate temporal patterns in landmark sequences.

**Advantages:**
- Processes sequential information.
- Learns temporal dependencies.
- Suitable for gesture-sequence modelling.

The GRU model has been trained and its model artifact has been saved. Its final test metrics are yet to be verified.

---

## 9. Model Evaluation Results

The trained Random Forest and SVM models were evaluated on the project's held-out test dataset.

### Performance Comparison

| Evaluation Metric | Random Forest | SVM |
|---|---:|---:|
| Accuracy | **80.4%** | 60.7% |
| Precision | 80.9% | 64.4% |
| Recall | 81.1% | 62.2% |
| F1-score | 78.2% | 61.1% |

### Accuracy Comparison

| Model | Test Accuracy |
|---|---:|
| Random Forest | 80.4% |
| SVM | 60.7% |

### Multimodal Feature Ablation Study

An additional experiment was conducted to investigate the effect of including body-pose information alongside hand landmarks.

| Feature Configuration | Accuracy |
|---|---:|
| Hand Landmarks Only | 78.5% |
| Hand Landmarks + Body Pose | **80.4%** |

**Key Observation:**

Combining hand landmarks with body-pose information improved test accuracy by 1.9 percentage points compared with using hand landmarks alone.

This experiment demonstrates the potential value of multimodal feature extraction for sign-language recognition.

*All reported metrics are from the actual training output. GRU performance will be added after its evaluation is verified.*

---

## 10. Project Directory Structure

```text
SignBridge-AI/
│
├── data/
│   └── Processed and dataset files
│
├── database/
│   └── Application database files
│
├── models/
│   ├── metrics.json
│   ├── labels.json
│   ├── static_model.pkl
│   ├── svm_model.pkl
│   └── dynamic_model/
│       └── gru.keras
│
├── scripts/
│   ├── extract_landmarks.py
│   ├── train_models.py
│   ├── list_labels.py
│   └── verify_setup.py
│
├── src/
│   ├── config.py
│   ├── hand_tracking.py
│   ├── feature_extraction.py
│   └── data_preprocessing.py
│
├── tests/
│
├── app.py
├── requirements.txt
└── README.md
```

---

## 11. Installation and Setup

### Prerequisites

- Windows, Linux or macOS.
- Python 3.11 (recommended for the current environment).
- pip package manager.
- Webcam for live application functionality.
- Git (optional).

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Navigate into the project:

```bash
cd SignBridge-AI
```

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```bash
python -m pip install --upgrade pip
```

```bash
pip install -r requirements.txt
```

### Step 4: Verify the Environment

```bash
python scripts/verify_setup.py
```

---

## 12. Running the Project

### Step 1: Test Landmark Extraction

Run a small test before processing the entire dataset:

```bash
python scripts/extract_landmarks.py --limit 5
```

If the test succeeds, process the remaining videos:

```bash
python scripts/extract_landmarks.py --workers 4
```

The extraction script supports resumable processing.

### Step 2: Train the Classical Models

```bash
python scripts/train_models.py --skip-gru
```

This trains and evaluates the Random Forest and SVM models.

### Step 3: Train All Models

```bash
python scripts/train_models.py
```

This runs the complete training pipeline, including GRU, provided the required dependencies and data are available.

### Step 4: Launch the Application

Once the application implementation is complete:

```bash
streamlit run app.py
```

---

## 13. Saved Model Artifacts

The trained model files are stored in the `models/` directory.

| File | Purpose |
|---|---|
| `static_model.pkl` | Saved Random Forest model |
| `svm_model.pkl` | Saved SVM model |
| `dynamic_model/gru.keras` | Saved GRU neural network |
| `labels.json` | Sign-class label mapping |
| `metrics.json` | Model evaluation information |

These saved artifacts can be loaded for prediction without retraining the models, provided the inference pipeline uses the same feature representation and preprocessing.

---

## 14. Applications and Real-World Impact

SignBridge-AI aims to contribute to more accessible communication through AI-based sign recognition.

Potential application areas include:

- **Education:** Supporting communication between students and educators.
- **Healthcare:** Exploring communication assistance in medical environments.
- **Public Services:** Improving accessibility in service centres.
- **Workplaces:** Supporting inclusive communication.
- **Everyday Interaction:** Making basic sign communication more accessible.

The current prototype recognises signs within its trained vocabulary. It is not a replacement for qualified sign-language interpreters.

---

## 15. Limitations

The current system has several limitations:

1. Recognition is restricted to the sign categories represented in the training dataset.
2. Prediction performance may vary with lighting, camera positioning and background conditions.
3. Different signing speeds and individual signing styles can affect recognition.
4. The current evaluation split is not signer-independent.
5. Isolated sign recognition does not automatically provide continuous sign-language translation.
6. Further testing with diverse signers and real-world environments is required.

---

## 16. Future Enhancements

The following improvements can extend SignBridge-AI:

- Continuous sign-language recognition.
- More extensive Indian Sign Language vocabulary.
- Transformer-based temporal recognition.
- Improved signer-independent evaluation.
- Multilingual speech output.
- Better prediction confidence calibration.
- Mobile application support.
- Real-world user testing.
- Cloud-based deployment.
- Accessibility-focused user interface improvements.

---

## 17. Screenshots and Demonstration

Add actual screenshots of the implemented application in this section.

Suggested screenshots:

| Screenshot | Description |
|---|---|
| `dashboard.png` | Main application dashboard |
| `translator.png` | Webcam-based sign recognition |
| `sentence-builder.png` | Text construction interface |
| `model-insights.png` | Model performance comparison |
| `history.png` | Translation history |

Suggested repository arrangement:

```text
assets/
└── screenshots/
    ├── dashboard.png
    ├── translator.png
    ├── sentence-builder.png
    ├── model-insights.png
    └── history.png
```

### Project Demonstration Video

Add your demonstration video link here after recording the working application.

**Demo:** [Add demonstration video link]

**Deployment:** [Add deployment link, if available]

---

## 18. Research Reference and Acknowledgement

This project draws on established research in Indian Sign Language recognition and computer vision.

### INCLUDE Dataset Research

Sridhar, A., Ganesan, R. G., Kumar, P., and Khapra, M. (2020).

*INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition.*

Proceedings of the ACM International Conference on Multimedia (MM '20).

DOI: https://doi.org/10.1145/3394171.3413528

The INCLUDE dataset and its associated research are acknowledged as relevant research references for Indian Sign Language recognition.

Repository: https://github.com/AI4Bharat/INCLUDE

Dataset: https://zenodo.org/record/4010759

**Note:** The INCLUDE research reference is not a claim that the current SignBridge-AI training results were obtained using its official benchmark split.

---

## 19. Responsible AI and Accessibility

SignBridge-AI is developed with the goal of improving accessibility and supporting inclusive communication.

The project recognises that sign languages have their own grammatical structures, regional variations and linguistic characteristics.

The current system should therefore be treated as an experimental recognition tool rather than a complete replacement for human interpretation.

Future development should involve feedback from sign-language users and accessibility experts.

---

## 20. Conclusion

SignBridge-AI demonstrates how artificial intelligence and computer vision can be combined to explore Indian Sign Language recognition.

Through landmark extraction, multimodal feature engineering and multiple machine learning architectures, the project establishes a foundation for an accessible sign-recognition system.

The project aims to move beyond gesture classification towards a practical communication-assistance platform that can help reduce communication barriers.

**SignBridge-AI – Technology that connects people beyond words.**

---

### Project Information

**Project:** SignBridge-AI  
**Domain:** Artificial Intelligence and Machine Learning  
**Focus:** Indian Sign Language Recognition  
**Language:** Python  
**Application Framework:** Streamlit  

**Developed as an AI/ML project focused on accessible communication.**
