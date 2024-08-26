# -*- coding: utf-8 -*-
"""
Created on Fri May 10 12:11:59 2024

@author: Madalin
"""

# %%
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import EarlyStopping

import random
random_seed=123
random.seed(random_seed)

from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.models import save_model, load_model

import mediapipe as mp

from tensorflow.keras.applications import VGG16

from math import ceil

file_path = os.path.dirname(__file__)
os.chdir(file_path)

train_dir = "asl_alphabet"

# %%

# Predictable signs:
classes = os.listdir(train_dir)
letter_to_label_dict = dict(zip(classes, range(len(classes))))
label_to_letter_dict = dict(zip(range(len(classes)), classes))

# %%

def create_path_df(train_dir):

    df = pd.DataFrame(columns=["path", "label"])
    for class_folder in os.listdir(train_dir):
        for class_image in os.listdir(os.path.join(train_dir, class_folder)):
            new_row = {"path": [os.path.join(train_dir, class_folder, class_image)], "label": [class_folder]}
            new_row = pd.DataFrame(new_row)
            df = pd.concat([df, new_row], ignore_index=True)
            # df.append(new_row, ignore_index=True)
            
    # randomly shuffle dataset
    df = df.sample(frac=1, random_state=random_seed)
    df.reset_index(inplace=True)
    return df
        
# %%
df = create_path_df(train_dir)

# %%

df['label'].value_counts()

# %%

def preprocess_image(img, shape):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, shape)
    img = img.astype(np.float64) / 255.0
    # img = img / 255.0
    return img
# %%

shape = (128, 128)

# %%

def canny_edge_segmentation(img, low_threshold=50, high_threshold=80, shape = (128, 128)):
    # Convert the image to grayscale if necessary
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    edges = cv2.Canny(img, low_threshold, high_threshold)
    
    # edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    
    mask = edges != 0
    segmented = img * (mask.astype(img.dtype))
    
    segmented = cv2.resize(segmented, shape)
    
    # Normalize and return the segmented image
    segmented = segmented.astype(np.float32) / 255.0
    segmented = np.expand_dims(segmented, axis=-1)  # Ensure shape is (H, W, 1)
    
    return segmented

# %%

img = cv2.imread(df.loc[3, 'path'])
#img = preprocess_image(img, shape)

# Visualize the original image
plt.figure(figsize=(6, 6))
plt.imshow(img, cmap='gray')  # Convert BGR to RGB for showing
plt.title("Original Image")
plt.axis('off')
plt.show()

# Apply Canny edge detection
segmented_img = canny_edge_segmentation(img, low_threshold=50, high_threshold=80)

# Visualize the segmented image
plt.figure(figsize=(6, 6))
plt.imshow(segmented_img[:, :, 0], cmap='gray')
plt.title("Segmented Image")
plt.axis('off')
plt.show()


# %%

import torch
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# Load a pre-trained MiDaS model
model_type = "MiDaS_small"  # or 'MiDaS_small' for less computational load
midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
midas.eval()

# Prepare the Midas transform (input normalization)
midas_transforms = Compose([
    Resize(384),
    ToTensor(),
    Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    ),
])

def estimate_depth(image_path):
    # Load and transform image
    img = Image.open(image_path).convert('RGB')
    img_input = midas_transforms(img).unsqueeze(0).to(device)
    
    # Predict and process depth
    with torch.no_grad():
        depth = midas(img_input)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=img.size[::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()
    
    # Normalize depth for visualization
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())
    return depth_normalized
# %%
image_path = df['path'][10]
image = cv2.imread(image_path)
depth_map = estimate_depth(image_path)


# Display the original image and the depth map
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(Image.open(image_path))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(depth_map, cmap='magma')
plt.title('Depth Map')
plt.axis('off')
plt.show()

# Apply simple thresholding for segmentation based on depth map
foreground_threshold = 0.6
depth_map = cv2.resize(depth_map, (128, 128), interpolation=cv2.INTER_NEAREST)
foreground_mask = depth_map > foreground_threshold

plt.figure()
plt.imshow(foreground_mask, cmap='gray')
plt.title('Foreground Segmentation')
plt.axis('off')
plt.show()

processed_img = preprocess_image(image, shape)
segmented_image = processed_img * foreground_mask

plt.figure()
plt.imshow(segmented_image, cmap='gray')
plt.title('Foreground Segmentation')
plt.axis('off')
plt.show()
# %%

for i, row in df.iterrows():
    img = cv2.imread(row['path'])
    plt.figure()
    plt.title(f"Letter: {row['label']}")
    segmented_img = canny_edge_segmentation(img, low_threshold=50, high_threshold=80, shape=(128, 128))
    plt.imshow(segmented_img, cmap='gray')
    
    if i > 3:
        break

# %%

for i, row in df.iterrows():
    print(row['path'])
    img = cv2.imread(row['path'])
    depth_map = estimate_depth(row['path'])
    foreground_threshold = 0.6
    depth_map = cv2.resize(depth_map, shape, interpolation=cv2.INTER_NEAREST)
    foreground_mask = depth_map > foreground_threshold
    processed_img = preprocess_image(img, shape)
    segmented_image = processed_img * foreground_mask
    plt.figure()
    plt.title(f"Letter: {row['label']}")
    
    plt.imshow(segmented_image, cmap='gray')
    
    if i > 10:
        break
# %%

shape = (128, 128)
# shape = (64, 64)

# %%
img = cv2.imread(df.loc[79, 'path'])
segmented_img = canny_edge_segmentation(img, low_threshold=50, high_threshold=100, shape=shape)
plt.imshow(segmented_img, cmap='gray')
# %%

img = cv2.imread(df.loc[1, 'path'])
img = preprocess_image(img, shape)
plt.imshow(img, cmap='gray')

# %%

def get_image_array_from_df(df):
    images = []
    for i, row in df.iterrows():
        img = cv2.imread(row['path'])
        # img = preprocess_image(img, shape)
        segmented_img = canny_edge_segmentation(img, low_threshold=50, high_threshold=80, shape=shape)
        images.append(segmented_img)
    
    images = np.array(images)
    return images

# %%

images = get_image_array_from_df(df)

# %%

def add_random_noise(img, factor = 0.05):
    noise_factor = factor * np.random.randn(*img.shape)
    img_noisy = img + noise_factor
    img_noisy = np.clip(img_noisy, 0, 1)
    return img_noisy

def manual_brightness_adjust(img, factor_range = (0.8, 1.2)):
    factor = np.random.uniform(factor_range[0], factor_range[1])
    adjusted_img = img * factor
    adjusted_img = np.clip(adjusted_img, 0, 1)
    return adjusted_img

def adjust_noise_and_brightness(img, noise_factor = 0.05, brightness_factor_range=(0.8, 1.2)):
    img = add_random_noise(img, noise_factor)
    img = manual_brightness_adjust(img, brightness_factor_range)
    return img

def erosion_dilation(img, erosion_iters=1, dilation_iters=1):
    if img.max() <= 1:
        img = (255 * img).astype(np.uint8)
    
    kernel = np.ones((3,3), np.uint8)
    
    img = cv2.erode(img, kernel, iterations=erosion_iters)
    img = cv2.dilate(img, kernel, iterations=dilation_iters)
    
    img = img.astype(np.float32) / 255.0
    
    return img

def apply_augmentations(img, noise_factor = 0.05, brightness_factor_range=(0.8, 1.2), erosion_iters=1, dilation_iters=1):
    # img = adjust_noise_and_brightness(img, noise_factor=noise_factor, brightness_factor_range=brightness_factor_range)
    # img = manual_brightness_adjust(img, brightness_factor_range)
    img = erosion_dilation(img, erosion_iters=erosion_iters, dilation_iters=dilation_iters)
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)
    return img

# %%

img = cv2.imread(df.loc[3, 'path'])
# img = preprocess_image(img, shape)
segmented_img = canny_edge_segmentation(img, low_threshold=50, high_threshold=100, shape=(128, 128))
img = apply_augmentations(img)
plt.imshow(segmented_img, cmap='gray', vmin=0, vmax=1)
# %%

train_datagen = ImageDataGenerator(
    rotation_range=5,
    # width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.2,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'#,
    # preprocessing_function=apply_augmentations
)

# %%

# img = cv2.imread(df.loc[4, 'path'])
img = images[35]
# img = canny_edge_segmentation(img, low_threshold=50, high_threshold=100, shape=(128, 128))
# img = add_random_noise(img)
# plt.imshow(img, cmap='gray')

# img = add_random_noise(img)

# img = np.expand_dims(img, -1)
img = np.expand_dims(img, 0)
# Using the generator as before
i = 0
for i, batch in enumerate(train_datagen.flow(img, batch_size=1)):
    # plt.subplot(1, 5, i + 1)
    plt.figure()
    plt.imshow(batch[0][:, :, 0], cmap='gray')
    plt.axis('off')
    if i >= 4:
        break

plt.show()


# %%

labels = df['label']

# Reshape the column to a 2D array (required by OneHotEncoder)
labels = np.array(labels).reshape(-1, 1)
encoder = OneHotEncoder(sparse_output=False)
labels = encoder.fit_transform(labels)
decoded_column = encoder.inverse_transform(labels)

# %%

X = images
y = labels

# %%

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed, stratify=y)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=random_seed, stratify=y_train)

# %%
# X_train = np.expand_dims(X_train, axis=-1)
# X_val = np.expand_dims(X_val, axis=-1)

batch_size = 32

# Ensure the data generator applies preprocessing and augmentation to each batch
train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)

# Data Generator for Validation/Test: No augmentation, only rescaling
test_datagen = ImageDataGenerator()

test_generator = test_datagen.flow(X_val, y_val, batch_size=batch_size)

# %%
num_classes = len(classes)

# %%
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))

optimizer = Adam(learning_rate=0.05)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# %%

from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D

base_model = ResNet50(include_top=False, weights=None, input_shape=(128, 128, 1))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# %%

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

input_tensor = Input(shape=(128, 128, 1))
base_model = EfficientNetB0(include_top=False, input_tensor=input_tensor, weights=None)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# %%

steps_per_epoch = ceil(len(X_train) / batch_size)
validation_steps = ceil(len(X_val) / batch_size)

early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=10,
    validation_data=test_generator,
    validation_steps=validation_steps,
    callbacks = [early_stopping]
)

# %%
img = X_test[1]
plt.imshow(img, cmap='gray')
img = np.expand_dims(img, -1)
img = np.expand_dims(img, 0)

# model.predict(img)
# np.argmax(model.predict(img))
label_to_letter_dict[np.argmax(model.predict(img))]
# %%
# X_test = np.expand_dims(X_test, -1)
model.evaluate(X_test, y_test)

# %%

mp_hands = mp.solutions.hands.Hands()
mpDraw = mp.solutions.drawing_utils

# %%

bbox_margin = 20
bbox_thickness = 2
cap = cv2.VideoCapture(0)
import matplotlib
while True:
    ret, frame = cap.read()
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_hands.process(frame)
    lmlist = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
           for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            # mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
            
            # Extract landmark coordinates
            landmarks_list = []
            for lm in hand_landmarks.landmark:
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks_list.append((cx, cy))
            
            # Calculate bounding box around the hand
            bbox = cv2.boundingRect(np.array(landmarks_list))
            x, y, w, h = bbox
            max_side = max(w, h)
            diff_x = max_side - w
            diff_y = max_side - h
            x -= diff_x // 2
            y -= diff_y // 2
            cv2.rectangle(frame, (x - bbox_margin, y - bbox_margin), (x + max_side + bbox_margin, y + max_side + bbox_margin), (0, 255, 0), 2)
            
            roi = frame[y - bbox_margin + bbox_thickness:y + max_side + bbox_margin -bbox_thickness, x - bbox_margin + bbox_thickness:x + max_side + bbox_margin - bbox_thickness]
            
            # roi = preprocess_image(roi, shape)
            roi = canny_edge_segmentation(roi, low_threshold=50, high_threshold=100, shape=(128, 128))
            # plt.imshow(roi, cmap='gray')
            # roi = np.expand_dims(roi, -1)
            roi = np.expand_dims(roi, 0)
            label = model.predict(roi)
            decoded_value = label_to_letter_dict[np.argmax(label)]
            
            cv2.putText(frame, decoded_value, (x, y + h + bbox_margin + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
            matplotlib.image.imsave('presentation.jpg', frame)

    
    # Display the resulting frame
    cv2.imshow('Skin Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

# %%

model.save('EfficientNetB0_2.keras')

# %%

model = load_model('EfficientNetB0_2.keras')