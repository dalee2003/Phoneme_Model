"""
Phoneme Prediction Test Script

This script loads our trained deep learning model and evaluates its performance on a subset of the TIMIT dataset that has 
been seen by the model yet. It preprocesses the test data (generate the mel spectrograms), feeds it into the model, 
and reports test accuracy and loss.

Requirements:
- TensorFlow/Keras for loading the trained model and running inference
- ImageDataGenerator for preprocessing and feeding images to the model
- NumPy for numerical operations
- Matplotlib for visualizing predictions

"""


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import datetime
from tensorflow.keras import optimizers

# Define test data directory
test_data_dir = 'timit_mel_images_test'  # Where test data's mel spectrograms are stored 

# Image dimensions
nrow, ncol = 200, 200 # for reshaping images to 200x200 pixels before being fed into the model 
batch_size_val = 5  # Batch size for test: how mnay images are processed at once 

# Load the trained model
model_path = "mdl_02_05_25_21_09_loss_0.67_acc_0.78.h5"  # file where trained model is stored
model = load_model(model_path) # loads the pre-trained model 

# Define ImageDataGenerator for test data (only rescaling is applied)
test_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0,
                                   zoom_range=0,
                                   horizontal_flip=False)

# Create test data generator: 
test_generator = test_datagen.flow_from_directory(
    test_data_dir, # Loads images from folders inside test_data_dir where each subfolder represents a different class label
    target_size=(nrow, ncol), # Resizes iamges to 200x200 pixels 
    batch_size=batch_size_val, # Uses batch size 5
    class_mode='sparse',  # Ensure class labels are integers
    shuffle=False
)

model.compile(optimizer=optimizers.Adam(learning_rate=0.0005),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Evaluate the model on test data: Determines how many batches are needed for one full pass over the test set.
test_steps = test_generator.n // batch_size_val # Number of total test images divided by batch size
loss, accuracy = model.evaluate(test_generator, steps=test_steps, verbose=1)

# Print test results
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")