<div align="center">

# 😃 Real-Time Facial Emotion Detection

**Detecting human emotions from a live webcam feed with CNNs and transfer learning**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-D00000?logo=keras&logoColor=white)](https://keras.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mahammadhajiiyev/emotion-detection-cnn/blob/main/Facial_emotion_recognition.ipynb)

</div>

---

## 📌 Overview

This project recognizes a person's facial emotion **in real time** through a webcam. It has two parts:

| Part | File | What it does |
|---|---|---|
| 🧠 **Training** | `Facial_emotion_recognition.ipynb` | Downloads the dataset, builds and trains two CNN models in Google Colab |
| 🎥 **Live detection** | `detection_emotion.py` | Finds faces in the webcam stream with OpenCV and shows the predicted emotion above each face |
| 📦 **Packaging** | `detection_emotion.spec` | PyInstaller config to turn the script into a standalone Windows `.exe` |

### Recognized emotions

| 😠 Angry | 😄 Happy | 😐 Neutral | 😢 Sad | 😲 Surprise |
|:---:|:---:|:---:|:---:|:---:|

## 📊 Dataset

| | |
|---|---|
| **Source** | [Kaggle — Emotion Recognition Dataset](https://www.kaggle.com/datasets/sujaykapadnis/emotion-recognition-dataset) |
| **Classes** | 5 — Angry, Happy, Neutral, Sad, Surprise |
| **Input size** | 224 × 224 RGB |

> The original dataset's `Ahegao` folder was removed, since it is not part of the emotion set covered by this project.

## 🔧 How it works

```mermaid
flowchart LR
    A[📷 Webcam frame] --> B[Haar Cascade<br/>face detection]
    B --> C[Crop & resize<br/>face to 224×224]
    C --> D[Trained CNN<br/>.keras model]
    D --> E[Emotion label<br/>drawn on screen]
```

## 🧠 Models

Two approaches are compared in the notebook:

| # | Model | Architecture |
|---|---|---|
| 1 | **Custom CNN** | Sequential `Conv2D` + `MaxPool2D` blocks, `he_normal` weight initialization |
| 2 | **ResNet50 transfer learning** | ImageNet-pretrained `ResNet50` (`include_top=False`) → `GlobalAveragePooling2D` → `Dense(5, softmax)`. Trained first with the base **frozen**, then **fine-tuned** with the base unfrozen |

Training setup for both models:

- **Loss:** `categorical_crossentropy` · **Optimizer:** `Adam` · **Metric:** `accuracy`
- `EarlyStopping` and `ModelCheckpoint` keep the best weights

## 🚀 Usage

### 1. Install requirements

```bash
pip install tensorflow opencv-python numpy matplotlib kaggle
```

### 2. Train the model

1. Click the **Open in Colab** badge above.
2. Upload your Kaggle API key (`kaggle.json`) — the dataset downloads and extracts automatically.
3. Run all cells. The trained model is saved in `.keras` format — download it to your computer.

### 3. Run live emotion detection

Update the model path in `detection_emotion.py` to where you saved the model:

```python
model = keras.models.load_model(r"your_model_path/emotions_predictions.keras")
```

Then start the script:

```bash
python detection_emotion.py
```

The webcam opens and the predicted emotion appears above each detected face. Press **`q`** to quit.

### 4. Build a standalone `.exe` (Windows)

```bash
pip install pyinstaller
pyinstaller detection_emotion.spec
```

The executable appears in the `dist/` folder.

## 📁 Project structure

```
emotion-detection-cnn/
├── Facial_emotion_recognition.ipynb   # dataset, preprocessing, model training
├── detection_emotion.py               # real-time webcam emotion detection
├── detection_emotion.spec             # PyInstaller build config
└── README.md
```

## ⚠️ Notes

- The model path in `detection_emotion.py` is hardcoded to a local Windows path — change it before running.
- The trained `.keras` model file is **not included** in the repo; generate it by running the notebook.

## 🔮 Future improvements

- Publish the trained model as a GitHub Release so the script works out of the box
- Replace the Haar Cascade with a more accurate face detector (MediaPipe, MTCNN)
- Add more emotions (Fear, Disgust) and a confidence score on screen
- Build a web demo with Streamlit or Gradio

## 👤 Author

**Mahammad Hajiyev** · [@mahammadhajiiyev](https://github.com/mahammadhajiiyev)
