import os
import cv2
import mediapipe as mp
import numpy as np
import time
from pymongo import MongoClient
import pandas as pd
from transformers import pipeline

# Suppress unnecessary TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize MongoDB client
client = MongoClient('mongodb://localhost:27017/')
db = client['gesture_keyboard_db']
collection = db['typed_text']

# Load word frequency dataset (optional, kept for fallback)
word_freq_df = pd.read_csv('~/unigram_freq.csv')
word_freq = dict(zip(word_freq_df['word'], word_freq_df['count']))
common_words = word_freq_df['word'].head(1000).tolist()

# Initialize Hugging Face pipeline for text generation
predictor = pipeline("text-generation", model="distilgpt2")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Keyboard Layout
keyboard_layout = [
    "1 2 3 4 5 6 7 8 9 0",
    "Q W E R T Y U I O P",
    "A S D F G H J K L",
    "Z X C V B N M",
    "SPACE BS ENTER"
]

# Key dimensions
KEY_WIDTH = 60
KEY_HEIGHT = 60
H_PADDING = 10
V_PADDING = 15
SPACE_KEY_WIDTH = 3 * KEY_WIDTH
BS_KEY_WIDTH = 2 * KEY_WIDTH
ENTER_KEY_WIDTH = 2 * KEY_WIDTH

start_x, start_y = 20, 20

output_text = ""
selection_threshold = 1.0
hover_start_time = {}
current_key = None
predicted_word = ""

def draw_keyboard(frame, fingertip_pos, window_width, window_height):
    global current_key
    current_key = None
    cx, cy = fingertip_pos

    rows = len(keyboard_layout)
    max_keys_in_row = max(len(row.split()) for row in keyboard_layout)
    key_width = min(KEY_WIDTH, (window_width - start_x * 2) // max_keys_in_row - H_PADDING)
    key_height = min(KEY_HEIGHT, (window_height - start_y * 2) // rows - V_PADDING)
    space_key_width = 3 * key_width
    bs_key_width = 2 * key_width
    enter_key_width = 2 * key_width

    y = start_y
    for row in keyboard_layout:
        x = start_x
        for key in row.split():
            key_width_dynamic = (
                space_key_width if key == "SPACE" else
                bs_key_width if key == "BS" else
                enter_key_width if key == "ENTER" else
                key_width
            )

            color = (200, 200, 200)
            if x <= cx <= x + key_width_dynamic and y <= cy <= y + key_height:
                color = (0, 255, 0)
                current_key = key

            cv2.rectangle(frame, (x, y), (x + key_width_dynamic, y + key_height), color, -1)
            text_x = x + key_width_dynamic // 3
            text_y = y + key_height // 2 + 10
            cv2.putText(frame, key, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

            x += key_width_dynamic + H_PADDING
        y += key_height + V_PADDING

    if predicted_word:
        cv2.putText(frame, f"Next: {predicted_word}", (start_x, y + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

def predict_next_word(current_text):
    """Predict the next word using Hugging Face pipeline."""
    if not current_text.strip():
        return "the"
    
    # Generate text with pipeline, adding 1 new token
    result = predictor(current_text.strip(), max_new_tokens=1, num_return_sequences=1, truncation=True)
    generated_text = result[0]['generated_text']
    
    # Extract the next word
    words = generated_text.split()
    if len(words) > len(current_text.split()):
        return words[-1]  # Return the last (new) word
    return "the"  # Fallback if no new word

def process_key_press(key):
    """Handles key press events, predicts next word on ENTER, and saves to MongoDB."""
    global output_text, predicted_word
    if key == "BS":
        output_text = output_text[:-1]
        predicted_word = ""
    elif key == "SPACE":
        output_text += " "
        predicted_word = ""
    elif key == "ENTER":
        predicted_word = predict_next_word(output_text)
        output_text += " " + predicted_word if predicted_word else ""
    else:
        output_text += key
        predicted_word = ""
    print("Output:", output_text)
    if predicted_word:
        print("Predicted:", predicted_word)
    
    doc = {
        "text": output_text,
        "timestamp": time.time(),
        "predicted_word": predicted_word if predicted_word else None
    }
    collection.insert_one(doc)

# Start Camera Capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    fingertip_pos = (0, 0)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            fingertip_pos = (cx, cy)

            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    draw_keyboard(frame, fingertip_pos, w, h)

    if current_key:
        if current_key not in hover_start_time:
            hover_start_time[current_key] = time.time()
        elif time.time() - hover_start_time[current_key] > selection_threshold:
            process_key_press(current_key)
            hover_start_time = {}
    else:
        hover_start_time = {}

    cv2.imshow("Gesture-Based Virtual Keyboard", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
client.close()