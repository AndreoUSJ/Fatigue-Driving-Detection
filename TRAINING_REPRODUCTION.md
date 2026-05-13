# Training Reproduction Instructions

This file explains how to reproduce the model training process used in this project.

The training process is based on the notebook:

```text
model_training.ipynb
```

The objective is to train and compare three fatigue-detection models:

1. Model 1, Simple CNN
2. Model 2, Deeper CNN
3. Model 3, MobileNetV2 transfer learning

Model 2 was selected because it achieved the highest DROWSY recall, which is the most important metric for this safety-oriented fatigue detection task.

## 1. Dataset Structure

The dataset must be organized into three folders:

```text
Drowsy_dataset/
├── train/
│   ├── DROWSY/
│   └── NATURAL/
├── val/
│   ├── DROWSY/
│   └── NATURAL/
└── test/
    ├── DROWSY/
    └── NATURAL/
```

Each class folder must contain the corresponding images.

The dataset used in the project was built from:

- Public fatigue-related datasets
- Custom in-car videos recorded by the team
- Cleaned frames after removing duplicate and near-duplicate images

## 2. Recommended Environment

The easiest way to reproduce training is by using Google Colab.

Recommended setup:

- Python 3
- TensorFlow / Keras
- NumPy
- Matplotlib
- scikit-learn
- Google Colab GPU runtime

Install required libraries if needed:

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

If running locally and testing live camera detection, also install:

```bash
pip install opencv-python mediapipe
```

## 3. Open the Notebook

Open the notebook in Google Colab:

```text
model_comp.ipynb
```

Mount Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Update the dataset path:

```python
base_path = "/content/drive/MyDrive/drive/Drowsy_dataset"
```

Make sure this path points to the folder containing:

```text
train/
val/
test/
```

## 4. Training Parameters

The training notebook uses the following parameters:

```python
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 20
```

Images are loaded in grayscale:

```python
color_mode='grayscale'
```

The task is binary classification:

```python
class_mode='binary'
```

## 5. Data Preprocessing and Augmentation

The training data uses:

```python
rescale=1./255
rotation_range=10
width_shift_range=0.1
height_shift_range=0.1
zoom_range=0.1
horizontal_flip=False
```

Validation and test images only use normalization:

```python
rescale=1./255
```

This means all pixel values are normalized to the range:

```text
[0, 1]
```

## 6. Models Trained

### Model 1, Simple CNN

This model is the baseline CNN.

Main structure:

```text
Input 128x128x1
Conv2D 32, 3x3, ReLU
MaxPooling
Conv2D 64, 3x3, ReLU
MaxPooling
Conv2D 128, 3x3, ReLU
MaxPooling
Flatten
Dense 128, ReLU
Dropout 0.3
Dense 1, Sigmoid
```

### Model 2, Deeper CNN

This model was selected as the final model.

Main structure:

```text
Input 128x128x1
Conv2D 32, 3x3, same padding, ReLU
BatchNormalization
MaxPooling

Conv2D 64, 3x3, same padding, ReLU
BatchNormalization
MaxPooling

Conv2D 128, 3x3, same padding, ReLU
BatchNormalization
MaxPooling

Conv2D 256, 3x3, same padding, ReLU
BatchNormalization
MaxPooling

Flatten
Dense 256, ReLU
Dropout 0.4
Dense 64, ReLU
Dropout 0.3
Dense 1, Sigmoid
```

### Model 3, MobileNetV2

MobileNetV2 was tested using transfer learning.

Since the dataset is grayscale, the single grayscale channel is copied three times to match the MobileNetV2 input format:

```text
128x128x1 -> 128x128x3
```

The MobileNetV2 base model uses ImageNet weights and is frozen during training.

## 7. Training Callbacks

The notebook uses three callbacks:

```python
EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)

ModelCheckpoint(f"{model_name}.keras", save_best_only=True, monitor='val_loss')
```

These callbacks:

- Stop training when validation loss stops improving
- Reduce the learning rate when needed
- Save the best model during training

## 8. Run Training

In the notebook, run:

```python
history_1 = train_model(model_1, "model_1_simple_cnn")
history_2 = train_model(model_2, "model_2_deeper_cnn")
history_3 = train_model(model_3, "model_3_mobilenet")
```

This trains the three models and saves the best `.keras` files.

Expected output files:

```text
model_1_simple_cnn.keras
model_2_deeper_cnn.keras
model_3_mobilenet.keras
```

## 9. Evaluate the Models

The notebook evaluates each model on the test set using:

- Accuracy
- Classification report
- Confusion matrix
- Test loss

Run:

```python
results_1 = evaluate_model(model_1, "Model 1 - Simple CNN")
results_2 = evaluate_model(model_2, "Model 2 - Deeper CNN")
results_3 = evaluate_model(model_3, "Model 3 - MobileNetV2")
```

Expected final comparison:

| Model | Accuracy | DROWSY Recall | Test Loss | Decision |
|---|---:|---:|---:|---|
| Model 1, Simple CNN | 80.99% | 93% | 0.8689 | Baseline |
| Model 2, Deeper CNN | 83.33% | 96% | 0.6451 | Selected |
| Model 3, MobileNetV2 | 83.85% | 83% | 0.3979 | Not selected |

Model 2 was selected because it achieved the best DROWSY recall.

In this project, DROWSY recall is more important than overall accuracy because missing a fatigued driver is the most dangerous error.

## 10. Save the Selected Model

The notebook saves the trained models to Google Drive.

Model 1:

```python
model_1.save("/content/drive/MyDrive/model_1_simple_cnn.keras")
```

Model 2:

```python
model_2.save("/content/drive/MyDrive/model_2_simple_cnn.keras")
```

Note: the original notebook saves Model 2 using the name:

```text
model_2_simple_cnn.keras
```

For clarity, it is recommended to rename it to:

```text
model_2_deeper_cnn.keras
```

## 11. Convert the Selected Model to TensorFlow Lite

After training, convert Model 2 to TensorFlow Lite:

```python
import tensorflow as tf

model = tf.keras.models.load_model("/content/drive/MyDrive/model_2_deeper_cnn.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("/content/drive/MyDrive/fatigue_model.tflite", "wb") as f:
    f.write(tflite_model)
```

The generated file is:

```text
fatigue_model.tflite
```

This file can be added to the Android application assets folder.

## 12. Testing the Trained Model Live

After training and saving the model, test it using the Python live testing script:

```bash
python live_test_model.py
```

On Windows, if multiple Python versions are installed, use:

```bash
py -3.11 live_test_model.py
```

This step is important because offline accuracy does not guarantee stable real-time behavior. In this project, Python live testing helped detect problems caused by blinking, yawning, hand movement, temporary face occlusion, and lighting changes before converting the model to the Android application.

## 13. Model Weights

The trained weights should be provided either in the repository or through an external download link.

Recommended repository paths:

```text
models/model_2_deeper_cnn.keras
app/src/main/assets/fatigue_model.tflite
```

If the files are too large for GitHub, upload them to Google Drive or GitHub Releases and add the links below:

```text
Keras model weights:
ADD_LINK_HERE

TensorFlow Lite model:
ADD_LINK_HERE
```

## 14. Important Notes

- The final selected model is Model 2, Deeper CNN.
- The input format is 128x128 grayscale.
- The output is a binary DROWSY probability.
- The model should not be used alone for the final fatigue decision.
- The final system combines CNN output with facial landmarks, calibration, and temporal validation.
- Python live testing should be done before Android integration.
