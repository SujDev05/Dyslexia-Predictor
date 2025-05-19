import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import image_dataset_from_directory, load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
import os

# ✅ Define dataset path
DATASET_PATH = "/Users/sujana/Documents/dyslexia/data"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 123

# ✅ Load training & validation datasets
train_dataset = image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,  # 80% training, 20% validation
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_dataset = image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# ✅ Class Names
class_names = train_dataset.class_names
print(f"Class names: {class_names}")  # Should print: ['dyslexic', 'non-dyslexic']

# ✅ Display Sample Images
plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()

# ✅ Optimize dataset loading
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)

# ✅ Build CNN Model
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(224, 224, 3)),  # Normalize images
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # Prevent overfitting
    layers.Dense(2, activation='softmax')  # Binary classification
])

# ✅ Compile the Model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ✅ Show Model Summary
model.summary()

# ✅ Train the Model
EPOCHS = 10

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS
)

# ✅ Evaluate Model
test_loss, test_acc = model.evaluate(val_dataset)
print(f"\n✅ Validation Accuracy: {test_acc * 100:.2f}%")
print(f"✅ Validation Loss: {test_loss:.4f}")

# ✅ Save Model
MODEL_PATH = "dyslexia_detection_model.h5"
model.save(MODEL_PATH)
print(f"✅ Model saved at {MODEL_PATH}")

# ✅ Function to Predict on a Single Image
def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return

    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions for model input

    prediction = model.predict(img_array)[0][0]  # Single probability output

    # Compute confidence scores
    dyslexic_confidence = prediction  # Probability of being dyslexic
    non_dyslexic_confidence = 1 - prediction  # Probability of being non-dyslexic

    # Determine the predicted class
    predicted_class = class_names[0] if dyslexic_confidence > non_dyslexic_confidence else class_names[1]

    print(f"\n✅ Dyslexic Confidence: {dyslexic_confidence:.4f}")
    print(f"✅ Non-Dyslexic Confidence: {non_dyslexic_confidence:.4f}")
    print(f"✅ Predicted Class: {predicted_class}")

# ✅ Test with a sample image
test_image = "/Users/sujana/Documents/dyslexia/data/non_dyslexic/5.jpg"
predict_image(test_image)