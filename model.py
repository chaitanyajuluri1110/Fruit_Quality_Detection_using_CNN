# train_model.py

import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Define the labels
labels = ['freshapples', 'rottenapples', 'freshbanana', 'rottenbanana']

# Data preprocessing using ImageDataGenerator
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Training and validation generators
tg = datagen.flow_from_directory(
    directory='C:/Users/DELL/Desktop/chaitanya/Train', 
    target_size=(20, 20), 
    classes=labels, 
    batch_size=25, 
    subset='training'
)

vg = datagen.flow_from_directory(
    directory='C:/Users/DELL/Desktop/chaitanya/Train', 
    target_size=(20, 20), 
    classes=labels, 
    batch_size=25, 
    subset='validation'
)

# Define the model
model = models.Sequential()
model.add(layers.Input(shape=(20, 20, 3)))
model.add(layers.Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    x=tg, 
    steps_per_epoch=len(tg), 
    epochs=8, 
    validation_data=vg, 
    validation_steps=len(vg)
)

# Save the model
model.save('fruit_quality_model.h5')

print("Model training complete and saved as 'fruit_quality_model.h5'")
