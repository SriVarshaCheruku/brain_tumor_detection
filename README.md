Brain Tumor Detection using Deep Learning (CNN)

This project uses Convolutional Neural Networks (CNNs) to detect brain tumors from MRI images. The model classifies images into two categories: **Tumor** and **No Tumor**, automating what would otherwise be a manual diagnostic process.

Abstract-

Brain tumor diagnosis through MRI imaging is critical but time-consuming and error-prone. This project proposes an automated method using deep learning to classify MRI scans with high accuracy, potentially assisting radiologists in faster and more accurate decision-making.

Objective-

To build and evaluate a CNN-based deep learning model that can automatically detect the presence of brain tumors in MRI images.


Dataset-

- Source: [Kaggle - Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- Structure:
braintumordataset/
├── yes/ # Tumor present
└── no/ # No tumor

Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy, Matplotlib
- Google Colab (for training)

Methodology

1. Data Preprocessing
- Images loaded from `yes/` and `no/` folders.
- Resized to **150x150** pixels.
- Normalized by dividing pixel values by 255.
- Dataset shuffled and split (80% training, 20% test).

2. CNN Model Architecture
- 3x Convolutional Layers with ReLU activation
- Max Pooling after each Conv layer
- Flattened output followed by Dense layer
- Dropout for regularization
- Sigmoid activation for binary classification

3. Compilation & Training
- Optimizer: Adam
- Loss: Binary Crossentropy
- Epochs: 10
- Batch Size: 32

Results

- Test Accuracy: ~84.31%
- 📉 Loss and accuracy plots saved:
- `accuracy_plot.png`
- `loss_plot.png`

Sample Training Output
Epoch 10/10
6/6 ━━━━━━━━━━━━━━━━━━━━ 0s 37ms/step - accuracy: 0.8680 - loss: 0.3289 - val_accuracy: 0.8293 - val_loss: 0.5406
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 248ms/step - accuracy: 0.8329 - loss: 0.3753
Test Accuracy: 84.31%

Training Graphs

Accuracy

![Accuracy](accuracy_plot.png)

Loss

![Loss](loss_plot.png)

How to Run

Option 1: Google Colab
1. Open the notebook in Colab.
2. Upload `braintumordataset.zip`.
3. Extract using:
   ```python
   !unzip braintumordataset.zip

