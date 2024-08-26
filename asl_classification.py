import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import EarlyStopping

import random
random_seed=123
random.seed(random_seed)

from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import save_model, load_model

import mediapipe as mp

from tensorflow.keras.applications import VGG16

file_path = os.path.dirname(__file__)
os.chdir(file_path)

train_dir = "asl_alphabet"
# %%

# Predictable signs:
classes = os.listdir(train_dir)

# Move the operators at the end of the list
operators = [4, 15, 21]
for operator_index in operators:
    classes.append(classes[operator_index])
    
for index, operator_index in enumerate(operators):
    del classes[operator_index]
    operators[(index+1) % (len(operators))] -= ((index+1) % (len(operators)))

#%%

letter_to_label_dict = dict(zip(classes, range(len(classes))))
label_to_letter_dict = dict(zip(range(len(classes)), classes))


# %%

df = pd.DataFrame(columns=["path", "label"])

# train_dataset = np.array()
for class_folder in os.listdir(train_dir):
    for class_image in os.listdir(os.path.join(train_dir, class_folder)):
        new_row = {"path": [os.path.join(train_dir, class_folder, class_image)], "label": [class_folder]}
        new_row = pd.DataFrame(new_row)
        df = pd.concat([df, new_row], ignore_index=True)
        # df.append(new_row, ignore_index=True)
        
# %%

df = df.sample(frac=1, random_state=random_seed)
df.reset_index(inplace=True)

# %%

df['label'].value_counts()

# %%

# %%
shape = (128, 128)

def preprocess_image(img, shape):
    img = cv2.resize(img, shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32) / 255.0
    return img

# %%

images = []
for _, row in df.iterrows():
    img = cv2.imread(row['path'])
    img = preprocess_image(img, shape)
    images.append(img)

# %%

images = np.array(images)

# %%

data_gen = ImageDataGenerator(
    rotation_range=25,      # Random rotations from -25 to 25 degrees
    width_shift_range=0.1,  # Random horizontal shifts +/- 10% of the width
    height_shift_range=0.1, # Random vertical shifts +/- 10% of the height
    shear_range=16,         # Shear angle in counter-clockwise direction in degrees
    zoom_range=0.25,        # Random zoom from 75% to 125%
    brightness_range=[0.8, 1.2], # Randomly adjust brightness between 80% and 120%
    horizontal_flip=True,   # Random horizontal flips
    fill_mode='nearest'     # Strategy to fill newly created pixels
)

# %%

img = np.expand_dims(img, -1)

img = np.expand_dims(img, 0)

# %%

img = cv2.imread(df['path'][1])
print(df['label'][1])
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img = cv2.resize(img, shape)

normalized_image = img.astype(np.float32) / 255.0

plt.imshow(img)

# %%

# img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# %%

labels = df['label']

# Reshape the column to a 2D array (required by OneHotEncoder)
labels = np.array(labels).reshape(-1, 1)

# Create the OneHotEncoder object
encoder = OneHotEncoder(sparse_output=False)

# Fit and transform the data
labels = encoder.fit_transform(labels)

decoded_column = encoder.inverse_transform(labels)
# %%

X = images
y = labels

# %%

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed, stratify=y)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=random_seed, stratify=y_train)

# %%

X_train = np.expand_dims(X_train, axis=-1)

# %%

train_generator = data_gen.flow(X_train[:3000], y_train[:3000], batch_size = 32)

# %%

# Sum along the appropriate axis to count values for each category
print(y_train.sum(axis=0))
print(y_test.sum(axis=0))
print(y_val.sum(axis=0))

# %%

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# %%
num_classes = 29
model = Sequential()
# Add convolutional layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

optimizer = Adam(learning_rate=0.1)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

model.summary()

# %%

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(128, 128, 1)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2), strides=(2, 2)),
    
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2), strides=(2, 2)),
    
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2), strides=(2, 2)),
    
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2), strides=(2, 2)),
    
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2), strides=(2, 2)),
    
    Flatten(),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')  # assuming 26 classes for ASL letters
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



# %%

history = model.fit(train_generator,
                    steps_per_epoch=3000 // 32,
                    epochs=10,
                    validation_data=(X_val, y_val),
                    callbacks=[early_stopping])

# %%
model.save('nn.keras')

# %%

model = load_model('nn.keras')

# %%

model.evaluate(X_test, y_test)

# %%

img = cv2.imread(df.loc[0, 'path'])
plt.imshow(img)
img = cv2.resize(img, shape)
img = img.reshape((1,64,64,3))
label = model.predict(img)
decoded_value = label_to_letter_dict[np.argmax(label)]
# %%

mp_hands = mp.solutions.hands.Hands()
frame = cv2.imread(df['path'][0])
plt.imshow(frame)
results = mp_hands.process(frame)

# %%

mpDraw = mp.solutions.drawing_utils

# %%

bbox_margin = 20

cap = cv2.VideoCapture(0)

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
            
            roi = frame[y - bbox_margin:y + max_side + 3 * bbox_margin, x - bbox_margin:x + max_side + bbox_margin]

            roi = cv2.resize(roi, shape)
            roi = roi.astype(np.float32) / 255.0
            plt.imshow(roi)
            roi = roi.reshape((1,64,64,3))
            
            label = model.predict(roi)
            decoded_value = label_to_letter_dict[np.argmax(label)]
            
            cv2.putText(frame, decoded_value, (x, y + h + bbox_margin + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    
    # Display the resulting frame
    cv2.imshow('Skin Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

# %%


