✋ ASL Sign Language Detection using MediaPipe & Random Forest

A lightweight, real-time hand sign recognition system that detects American Sign Language (ASL) gestures (A–Z) using [MediaPipe](https://google.github.io/mediapipe/) and a Random Forest classifier. Trained on custom-collected webcam data.

🎥 Demo

> 🔗 [Click here to watch the demo video](https://youtu.be/CVvE1yCH0TQ)  
> (Uploaded on YouTube)

🚀 Features
- ✅ Collects webcam images for all 26 alphabets (A–Z)
- ✅ Extracts 21 landmark keypoints per hand using MediaPipe
- ✅ Trains a scikit-learn Random Forest classifier on landmarks
- ✅ Real-time prediction with accuracy score display
- ✅ Generates confusion matrix to visualize performance

📁 Project Structure
--> collect_all_alphabets.py # Webcam-based data collection

-->extract_landmarks.py # Extract hand keypoints using MediaPipe

-->train_model.py # Train Random Forest model

-->inference.py # Real-time detection from webcam

-->data/ # Collected images per class (A–Z)

-->model.p # Trained model + label encoder

-->data.pickle # Landmark data and labels

-->confusion_matrix.png # Evaluation output
