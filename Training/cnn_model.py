"""
Phoneme Prediction CNN Model Training

This script trains a Convolutional Neural Network (CNN) for phoneme classification using image representations 
of preprocessed speech data (Mel Spectrograms). It uses the preprocessed the TIMIT dataset mel spectrogram images 
to train the model with early stopping to optimize performance. The trained model is then saved with a filename 
indicating its best validation loss and accuracy.

Requirements:
- TensorFlow/Keras
- NumPy
- Matplotlib
- datetime
"""



from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import datetime

# Set the input shape for the images: 200x200 pixels with 3 color channels (RGB)
nrow = 200
ncol = 200
input_shape = (nrow, ncol, 3)

# Clear any previous sessions to avoid memory issues when creating a new model
K.clear_session()

# Create a Sequential model (used for stacking layers in a linear fashion)
model = Sequential()

# Add the first convolutional layer with 16 filters, 7x7 kernel size, ReLU activation, and same padding
model.add(Conv2D(16, (7, 7), activation='relu', padding='same', input_shape=input_shape))
# Add max pooling layer with 3x3 pool size and stride of 2
model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
# Add batch normalization to stabilize and accelerate training
model.add(BatchNormalization())

# Add a second convolutional layer with 32 filters, 5x5 kernel size, ReLU activation, and same padding
model.add(Conv2D(32, (5, 5), activation='relu', padding='same'))
# Add max pooling and batch normalization as done in previous layers
model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
model.add(BatchNormalization())

# Add a third convolutional layer with 64 filters, 3x3 kernel size, ReLU activation, and same padding
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
# Max pooling and batch normalization for this layer too
model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
model.add(BatchNormalization())

# Add a fourth convolutional layer with 128 filters, 3x3 kernel size, ReLU activation, and same padding
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
model.add(BatchNormalization())

# Add a fifth convolutional layer with 128 filters, 3x3 kernel size, ReLU activation, and same padding
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
model.add(BatchNormalization())

# Add a sixth convolutional layer with 256 filters, 3x3 kernel size, ReLU activation, and same padding
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
model.add(BatchNormalization())

# Flatten the output from the convolutional layers to feed into the fully connected layers
model.add(Flatten())
# Add a fully connected layer with 1024 neurons and ReLU activation
model.add(Dense(1024, activation='relu'))
# Add batch normalization to improve convergence
model.add(BatchNormalization())
# Add dropout layer to prevent overfitting (50% chance of dropping units during training)
model.add(Dropout(0.5))

# Output layer with 61 neurons (one for each phoneme class) and softmax activation for multi-class classification
model.add(Dense(61, activation='softmax'))

# Print a summary of the model architecture
model.summary()

# Define the directory containing the training images
train_data_dir = 'timit_mel_images'
batch_size_tr = 32  # Batch size for training

# ImageDataGenerator is used to preprocess the images and perform data augmentation (though here no augmentation is used)
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0,
                                   zoom_range=0,
                                   horizontal_flip=False)

# Create a generator to load images from the training directory
train_generator = train_datagen.flow_from_directory(
                        train_data_dir,         # Path to the training data directory
                        target_size=(nrow, ncol),  # Resize images to 200x200
                        batch_size=batch_size_tr,  # Batch size
                        class_mode='sparse')  # Use sparse labels (integers)

# Define the directory containing the test images
test_data_dir = 'timit_mel_images_test'

batch_size_ts = 5  # Batch size for testing

# Similarly, define the ImageDataGenerator for test images (only rescaling is applied here)
test_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0,
                                   zoom_range=0,
                                   horizontal_flip=False)

# Create a generator for loading images from the test directory
test_generator = test_datagen.flow_from_directory(
                        test_data_dir,              # Path to the test data directory
                        target_size=(nrow, ncol),   # Resize test images to 200x200
                        batch_size=batch_size_ts,   # Batch size for testing
                        class_mode='sparse')        # Use sparse labels (integers)

# Compile the model with Adam optimizer, sparse categorical cross-entropy loss, and accuracy as the metric
model.compile(optimizer=optimizers.Adam(learning_rate=0.0005),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Define steps per epoch and validation steps based on the number of samples and batch sizes
steps_per_epoch =  train_generator.n // batch_size_tr
validation_steps =  test_generator.n // batch_size_ts

# Set the number of epochs to train the model
nepochs = 100  # Number of epochs

# Early stopping to stop training if the validation loss doesn't improve for 5 epochs
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 5, restore_best_weights = True)

# Train the model with the training and validation data
history = model.fit(
    train_generator,                    # Training data generator
    steps_per_epoch=steps_per_epoch,    # Number of steps per epoch
    epochs=nepochs,                     # Number of epochs to train
    validation_data=test_generator,     # Validation data generator
    validation_steps=validation_steps,  # Number of steps for validation
    callbacks=([early_stopping])        # Early stopping callback to prevent overfitting
)

# Extract the best loss and accuracy from the training history
best_loss = min(history.history['val_loss'])
best_acc = max(history.history['val_accuracy'])

# Get the current timestamp to use in the model's filename
timestamp = datetime.datetime.now().strftime("%m_%d_%y_%H_%M")

# Save the trained model with a filename that includes the timestamp, best loss, and accuracy
model_filename = f"mdl_{timestamp}_loss_{best_loss:.2f}_acc_{best_acc:.2f}.h5"
model.save(model_filename)