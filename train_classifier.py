import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# === Load landmark data ===
with open('./data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# === Encode labels A-Z to integers ===
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)  # 'A' => 0, 'B' => 1, ...

# === Train-test split ===
x_train, x_test, y_train, y_test = train_test_split(
    data, labels_encoded, test_size=0.2, shuffle=True, stratify=labels_encoded
)

# === Train model ===
model = RandomForestClassifier()
model.fit(x_train, y_train)

# === Evaluate ===
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"{accuracy * 100:.2f}% of samples were classified correctly!")

# === Save model and label encoder ===
with open('model.p', 'wb') as f:
    pickle.dump({
        'model': model,
        'label_encoder': label_encoder
    }, f)

print("Model and label encoder saved to 'model.p'")

# === Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
labels_list = label_encoder.classes_  # ['A', 'B', ..., 'Z']

plt.figure(figsize=(16, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels_list,
            yticklabels=labels_list)

plt.title('Confusion Matrix for ASL Sign Classifier', fontsize=18)
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)
plt.tight_layout()
plt.savefig("confusion_matrix.png")  # Save to file for LinkedIn
plt.show()