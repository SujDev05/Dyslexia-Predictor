
# Dyslexia Detection Using CNN

This project implements a Convolutional Neural Network (CNN) to classify images into **dyslexic** and **non-dyslexic** categories using TensorFlow and Keras. It loads image data from a structured directory, trains a CNN model, evaluates its performance, and provides a function to predict the class of new images.

---

## Features

- Loads dataset from local directory with an 80-20 train-validation split.
- Displays sample images from the training set.
- Builds a CNN model with multiple convolutional and pooling layers.
- Normalizes input images and uses dropout for regularization.
- Trains and evaluates the model, showing accuracy and loss.
- Saves the trained model as `dyslexia_detection_model.h5`.
- Includes a prediction function for single image inference.

---

## Project Structure

```
/data
  /dyslexic
    - img1.jpg
    - img2.jpg
    ...
  /non-dyslexic
    - img1.jpg
    - img2.jpg
    ...
training.py         # Main training and prediction script
dyslexia_detection_model.h5   # Saved trained model (excluded from repo if large)
README.md
```

---

## Setup & Run Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/SujDev05/Dyslexia-Predictor.git
cd Dyslexia-Predictor
```

### 2. Set Up a Virtual Environment (Optional but Recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install tensorflow matplotlib numpy 
```
NOTE : use tensorflow if your a Mac user with silicon-chip . 
       if you are a windows user , install torchvision

### 4. Prepare the Dataset

- Organize your images in the `data` directory with subfolders:
  - `data/dyslexic/`
  - `data/non-dyslexic/`
- Place the corresponding images in each folder.

### 5. Update Dataset Path

- Open `training.py`
- Modify the `DATASET_PATH` variable to the absolute path of your dataset directory.  
  Example:  
  ```python
  DATASET_PATH = "/Users/sujana/Documents/dyslexia/data"
  ```

### 6. Run the Training Script

```bash
python training.py
```

- This will train the CNN model for 10 epochs.
- Sample images from the training set will be displayed.
- The model summary and training progress will print to the console.
- The trained model will be saved as `dyslexia_detection_model.h5`.

### 7. Predict on a New Image

- Use the `predict_image(image_path)` function inside `training.py` to classify a new image.
- Example usage inside the script:

```python
test_image = "/path/to/your/test/image.jpg"
predict_image(test_image)
```

---

## Notes

- Images are resized to 224x224 pixels.
- The model uses softmax activation to output class probabilities.
- Large model files (`.h5`) are excluded from the repo to avoid GitHub size limits.But when you run the model , it will be saved with .h5 extension 
- For large files, consider using Git Large File Storage (Git LFS).

---

## License

This project is open source and available under the MIT License.

---

## Author

Sujana S
