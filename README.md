# Sinhala Handwritten Character Recognition

A machine learning project for recognizing handwritten Sinhala characters using K-Nearest Neighbors (KNN) algorithm.

## Project Overview

This project implements a system that can recognize handwritten Sinhala characters. The system uses a K-Nearest Neighbors (KNN) algorithm trained on a dataset of Sinhala characters. Users can draw characters on a graphical interface, and the system will predict which Sinhala character was drawn.

## Features

- Dataset creation and preprocessing from image files
- Training a KNN classifier on the prepared dataset
- Interactive GUI for drawing characters and getting real-time predictions
- Saving drawn characters for further dataset expansion
- High accuracy recognition of 4 Sinhala characters (අ, එ, ඉ, උ)

## Repository Contents

- `1-dataset-creation.ipynb`: Jupyter notebook for creating and preprocessing the dataset
- `2-training-the-KNN.ipynb`: Jupyter notebook for training the KNN classifier
- `3-sinhala-character-gui.ipynb`: Jupyter notebook with GUI implementation for character recognition
- `GUI.ipynb`: Alternative GUI implementation
- `data.npy`: Numpy array containing processed image data
- `target.npy`: Numpy array containing target labels
- `sinhala-character-knn.sav`: Saved KNN model
- `dataset/`: Directory containing raw image files for training
  - `a/`: Images of Sinhala character 'අ'
  - `ae/`: Images of Sinhala character 'එ'
  - `e/`: Images of Sinhala character 'ඉ'
  - `u/`: Images of Sinhala character 'උ'
- `data/`: Directory for saving new drawings from the GUI

## Technical Details

### Dataset Preparation
- Images are resized to 8×8 pixels
- Images are converted to grayscale
- Data is flattened to 64-dimensional vectors
- Dataset contains 270 samples across 4 character classes

### Model Training
- K-Nearest Neighbors algorithm (default parameters)
- 80% training / 20% test split
- Achieved accuracy of approximately 80%
- Model is serialized using joblib for later use

### GUI Implementation
- Built with Tkinter
- Canvas for drawing characters
- Buttons for predicting, saving, clearing, and exiting
- Real-time prediction display

## Dependencies

- Python 3.x
- NumPy
- Matplotlib
- OpenCV (cv2)
- scikit-learn
- Tkinter
- PIL (Pillow)
- joblib

## Usage Instructions

### 1. Dataset Creation
Run the `1-dataset-creation.ipynb` notebook to process the raw image files and create the training dataset.

```python
# Key operations:
# - Load images from dataset folders
# - Resize images to 8×8
# - Convert to grayscale
# - Reshape to 1D array (64 features)
# - Save processed data as numpy arrays
```

### 2. Training the Model
Run the `2-training-the-KNN.ipynb` notebook to train the KNN model on the prepared dataset.

```python
# Key operations:
# - Load the data and target arrays
# - Split into train and test sets
# - Initialize and train the KNN classifier
# - Evaluate model performance
# - Save the trained model
```

### 3. Using the GUI
Run either `3-sinhala-character-gui.ipynb` or `GUI.ipynb` to launch the interactive interface.

- Draw a Sinhala character in the canvas area
- Click the "PREDICT" button to see the recognition result
- Click "SAVE" to store the drawing for future training
- Click "CLEAR" to reset the canvas
- Click "EXIT" to close the application

## Future Enhancements

- Expand the dataset to include more Sinhala characters
- Implement more advanced models (CNN, LSTM)
- Add data augmentation for improved robustness
- Develop a standalone application
- Add support for recognizing multiple characters or words

## Author

Rensith Udara

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- Special thanks to the contributors who helped with data collection
- Inspired by similar handwritten digit recognition systems like MNIST