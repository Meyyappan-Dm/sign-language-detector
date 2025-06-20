import cv2
import os
import time
import string

# === CONFIGURATION ===
num_images_per_class = 100      # Number of images per letter
capture_delay = 0.05            # Time delay between captures (seconds)
camera_index = 0                # Change if camera 0 fails

# === SETUP ===
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print("Error: Cannot open webcam.")
    exit()

output_base_dir = "data"
os.makedirs(output_base_dir, exist_ok=True)

# === ALPHABET LOOP ===
for letter in string.ascii_uppercase:
    print(f"\nGet ready to show sign for: '{letter}'")
    for i in range(3, 0, -1):
        print(f"Starting in {i}...", end='\r')
        time.sleep(1)

    class_dir = os.path.join(output_base_dir, letter)
    os.makedirs(class_dir, exist_ok=True)
    count = 0

    while count < num_images_per_class:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        cv2.putText(frame, f"Class: {letter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Images: {count}/{num_images_per_class}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow('Data Collection', frame)

        filename = os.path.join(class_dir, f"{count}.jpg")
        cv2.imwrite(filename, frame)
        count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quit requested by user.")
            cap.release()
            cv2.destroyAllWindows()
            exit()

        time.sleep(capture_delay)

    print(f"Collected {num_images_per_class} images for '{letter}'")

print("\nData collection complete for all alphabets!")
cap.release()
cv2.destroyAllWindows()

