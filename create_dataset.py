import os
import pickle
import mediapipe as mp
import cv2
import string

# === Mediapipe Setup ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# === Paths ===
DATA_DIR = './data'
OUTPUT_FILE = 'data.pickle'

# === Storage ===
data = []
labels = []

# === Loop over A–Z folders only ===
for dir_ in sorted(os.listdir(DATA_DIR)):
    if dir_ not in string.ascii_uppercase:
        continue  # Skip folders that aren't A–Z

    print(f"Processing letter: {dir_}")

    for img_file in os.listdir(os.path.join(DATA_DIR, dir_)):
        img_path = os.path.join(DATA_DIR, dir_, img_file)

        # Read and preprocess image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Skipped invalid image: {img_path}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        # Extract landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                data_aux = []
                x_ = []
                y_ = []

                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)

                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))

                data.append(data_aux)
                labels.append(dir_)

# === Save to pickle ===
with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"\n Landmark data extracted and saved to '{OUTPUT_FILE}'")
