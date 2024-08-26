# -*- coding: utf-8 -*-
"""
Created on Fri May 10 11:00:43 2024

@author: Madalin
"""

# %%

import os
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import mediapipe as mp

import random
random_seed=123
random.seed(random_seed)

file_path = os.path.dirname(__file__)
os.chdir(file_path)

# %%

shape = (128,128)

# %%

img = cv2.imread('asl_alphabet_train\\asl_alphabet_train\\D\\D1.jpg')
plt.imshow(img)
letter = 'Z'
class_counter = 0
letter_dir = 'asl_alphabet/' + letter

# %%
mp_hands = mp.solutions.hands.Hands()
mpDraw = mp.solutions.drawing_utils

bbox_margin = 20
bbox_thickness = 2
cap = cv2.VideoCapture(0)
capture_hand = False

while True:
    ret, frame = cap.read()
    
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
            cv2.rectangle(frame, (x - bbox_margin, y - bbox_margin), (x + max_side + bbox_margin, y + max_side + bbox_margin), (0, 255, 0), bbox_thickness)
            
            roi = frame[y - bbox_margin + bbox_thickness:y + max_side + bbox_margin -bbox_thickness, x - bbox_margin + bbox_thickness:x + max_side + bbox_margin - bbox_thickness]

            # roi = cv2.resize(roi, shape)
            # roi = roi.astype(np.float32) / 255.0
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            if capture_hand:
                matplotlib.image.imsave(letter_dir + '/' + letter + str(class_counter) +'.jpg', roi)
                class_counter += 1
                
            
            # roi = roi.reshape((1,64,64,3))
            
            # label = model.predict(roi)
            # decoded_value = label_to_letter_dict[np.argmax(label)]
            
            # cv2.putText(frame, decoded_value, (x, y + h + bbox_margin + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    
    # Display the resulting frame
    cv2.imshow('Skin Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    if cv2.waitKey(1) & 0xFF == ord('s'):
        capture_hand = not capture_hand
        print("Flag set to " + str(capture_hand))

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


# %%