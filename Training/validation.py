"""
Phoneme Prediction Validation Script

This script loads our trained deep learning model and evaluates its performance on a subset of the TIMIT dataset that has 
been seen by the model yet. It preprocesses the validation data (generate the mel spectrograms), feeds it into the model, 
and reports validation accuracy and loss.

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

# Define validation data directory
validation_data_dir = 'timit_mel_images_validation'  # Where validation data's mel spectrograms are stored 

# Image dimensions
nrow, ncol = 200, 200 # for reshaping images to 200x200 pixels before being fed into the model 
batch_size_val = 5  # Batch size for validation: how mnay images are processed at once 

# Load the trained model
model_path = "mdl_02_03_25_20_20_loss_0.70_acc_0.78.h5"  # file where trained model is stored
model = load_model(model_path) # loads the pre-trained model 

# Define ImageDataGenerator for validation data (only rescaling is applied)
val_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0,
                                   zoom_range=0,
                                   horizontal_flip=False)

# Create validation data generator: 
val_generator = val_datagen.flow_from_directory(
    validation_data_dir, # Loads images from folders inside validation_data_dir where each subfolder represents a different class label
    target_size=(nrow, ncol), # Resizes iamges to 200x200 pixels 
    batch_size=batch_size_val, # Uses batch size 5
    class_mode='sparse',  # Ensure class labels are integers
    shuffle=False
)

model.compile(optimizer=optimizers.Adam(learning_rate=0.0005),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Evaluate the model on validation data: Determines how many batches are needed for one full pass over the validation set.
validation_steps = val_generator.n // batch_size_val # Number of total validation images divided by batch size
loss, accuracy = model.evaluate(val_generator, steps=validation_steps, verbose=1)

# Print validation results
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")