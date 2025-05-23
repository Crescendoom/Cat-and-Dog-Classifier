"""
 Here's the code I used if you want to train your own model
"""
# import os
# import cv2
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
# from tensorflow.keras.applications import MobileNetV2 # type: ignore
# from tensorflow.keras.models import Model # type: ignore
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout # type: ignore
# from tensorflow.keras.optimizers import Adam # type: ignore
# from tensorflow.keras.callbacks import EarlyStopping # type: ignore
# from glob import glob
# # from google.colab import drive
# # from google.colab import files
# plt.style.use('ggplot')

"""
It depends where is the directory of your dataset
If you are using Google Colab, uncomment the lines below to mount your Google Drive
"""
# Mount Google Drive
# drive.mount('/content/drive')

# base_dir = "Dataset"
# train_dir = os.path.join(base_dir, "training_set")
# val_dir = os.path.join(base_dir, "validation_set")
# cat_files = glob(os.path.join(train_dir, "cats", "*.jpg"))
# dog_files = glob(os.path.join(train_dir, "dogs", "*.jpg"))

# target_size = (224, 224)
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=30,
#     zoom_range=0.2,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     shear_range=0.2,
#     horizontal_flip=True
# )

# val_datagen = ImageDataGenerator(rescale=1./255)
# train_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=target_size,
#     batch_size=32,
#     class_mode='binary'
# )
# val_generator = val_datagen.flow_from_directory(
#     val_dir,
#     target_size=target_size,
#     batch_size=32,
#     class_mode='binary'
# )

# base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
# base_model.trainable = False
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dropout(0.5)(x)
# x = Dense(128, activation='relu')(x)
# predictions = Dense(1, activation='sigmoid')(x)

# model = Model(inputs=base_model.input, outputs=predictions)
# model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
# history = model.fit(
#     train_generator,
#     validation_data=val_generator,
#     epochs=10,
#     callbacks=[early_stop])

"""
Code block for plotting the accuracy and loss of your training
You can uncomment the lines below if you want to plot the accuracy and loss of your training
"""
# # PLOT THE ACCURACY AND LOSS OF YOUR TRAINING
# # plt.plot(history.history['accuracy'], label='Train Accuracy')
# # plt.plot(history.history['val_accuracy'], label='Val Accuracy')
# # plt.title('Model Accuracy')
# # plt.xlabel('Epoch')
# # plt.ylabel('Accuracy')
# # plt.legend()
# # plt.show()
# # plt.plot(history.history['loss'], label='Train Loss')
# # plt.plot(history.history['val_loss'], label='Val Loss')
# # plt.title('Model Loss')
# # plt.xlabel('Epoch')
# # plt.ylabel('Loss')
# # plt.legend()
# # plt.show()

"""
Code block for saving and loading the model
You can save the model with ".keras" or ".h5" extension
"""
# model.save("catAndDog_BinaryClassifier.keras")
# model.save_weights("model_weights.keras")
# model = create_model()
# model.load_weights("model_weights.keras")
