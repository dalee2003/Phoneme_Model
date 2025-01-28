#Creating the Model

from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping


nrow = 200
ncol = 200
input_shape = (nrow, ncol, 3)
K.clear_session()

# Create a new model
model = Sequential()

model.add(Conv2D(16, (7, 7), activation='relu', padding='same', input_shape=input_shape))
model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
model.add(BatchNormalization())

model.add(Conv2D(32, (5, 5), activation='relu', padding='same', input_shape=input_shape))
model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape))
model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
model.add(BatchNormalization())

model.add(Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=input_shape))
model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
model.add(BatchNormalization())

model.add(Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=input_shape))
model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
model.add(BatchNormalization())

model.add(Conv2D(256, (3, 3), activation='relu', padding='same', input_shape=input_shape))
model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(61, activation='softmax'))

model.summary()


train_data_dir = 'timit_mfcc_images'
batch_size_tr = 32
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0,
                                   zoom_range=0,
                                   horizontal_flip=False)
train_generator = train_datagen.flow_from_directory(
                        train_data_dir,
                        target_size=(nrow,ncol),
                        batch_size=batch_size_tr,
                        class_mode='sparse')

test_data_dir = 'timit_mfcc_images_test'
batch_size_ts = 5
test_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0,
                                   zoom_range=0,
                                   horizontal_flip=False)
test_generator = test_datagen.flow_from_directory(
                        test_data_dir,
                        target_size=(nrow,ncol),
                        batch_size=batch_size_ts,
                        class_mode='sparse')

model.compile(optimizer=optimizers.Adam(learning_rate=0.0005),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


steps_per_epoch =  train_generator.n // batch_size_tr
validation_steps =  test_generator.n // batch_size_ts

nepochs = 100  # Number of epochs

early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=nepochs,
    validation_data=test_generator,
    validation_steps=validation_steps,
    callbacks=([early_stopping])
)

# Save the model
model.save('phonemes_mfcc_cnn_model.h5')
print("\nModel saved\n")



###############CONFUSION MATRIX#################
# import numpy as np
# from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# # Load the saved model
# model = load_model('val_loss_0.46_val_acc_0.87_mfcc.h5')

# # Define the parameters (make sure these match your original settings)
# nrow = 200
# ncol = 200
# #test_data_dir = './timit_mfcc_images_test'
# test_data_dir = './timit_mfcc_images'
# batch_size_ts = 5

# # Create the test data generator
# test_datagen = ImageDataGenerator(rescale=1./255,
#                                    shear_range=0,
#                                    zoom_range=0,
#                                    horizontal_flip=False)
# test_generator = test_datagen.flow_from_directory(
#                         test_data_dir,
#                         target_size=(nrow,ncol),
#                         batch_size=batch_size_ts,
#                         class_mode='sparse',
#                         shuffle=False)  # Important: set shuffle to False

# # Calculate steps
# validation_steps = test_generator.n // batch_size_ts

# # Generate predictions
# test_generator.reset()
# y_true = test_generator.classes
# y_pred = model.predict(test_generator, steps=validation_steps)
# y_pred_classes = np.argmax(y_pred, axis=1)

# # Create the confusion matrix
# cm = confusion_matrix(y_true[:len(y_pred_classes)], y_pred_classes)

# # Visualize the confusion matrix
# plt.figure(figsize=(20,20))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.title('Confusion Matrix')
# plt.show()

# # Optional: Normalize the confusion matrix
# cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# plt.figure(figsize=(20,20))
# sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues')
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.title('Normalized Confusion Matrix')
# plt.show()
