import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
from collections import Counter

# Load your dataset
def load_data(directory, label=None):
    import fitz
    import pdfplumber

    texts, labels = [], []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.pdf'):
                try:
                    with fitz.open(os.path.join(root, file)) as pdf:
                        text = "".join([page.get_text() for page in pdf])
                except:
                    with pdfplumber.open(os.path.join(root, file)) as pdf:
                        text = " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])
                if text.strip():
                    texts.append(text.strip())
                    if label is not None:
                        labels.append(label)
    return texts, labels

# Directories
publishable_dir = "/Users/sathishm/Downloads/Publishable"
non_publishable_dir = "/Users/sathishm/Downloads/Non-Publishable"

# Load data
texts_publishable, labels_publishable = load_data(publishable_dir, label="Publishable")
texts_non_publishable, labels_non_publishable = load_data(non_publishable_dir, label="Non-Publishable")

texts = texts_publishable + texts_non_publishable
labels = labels_publishable + labels_non_publishable

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
joblib.dump(label_encoder, "label_encoder.pkl")  # Save label encoder

# Print class distribution
print("Class distribution before SMOTE:", Counter(labels_encoded))

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
X = vectorizer.fit_transform(texts).toarray()
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")  # Save TF-IDF vectorizer

# Oversample data using SMOTE
min_samples_per_class = min(np.bincount(labels_encoded))
n_neighbors = min(5, min_samples_per_class - 1)  # Ensure n_neighbors <= min_samples_per_class - 1
smote = SMOTE(random_state=42, k_neighbors=n_neighbors)

X_resampled, y_resampled = smote.fit_resample(X, labels_encoded)
y_resampled = to_categorical(y_resampled)

# Print class distribution after SMOTE
print("Class distribution after SMOTE:", Counter(np.argmax(y_resampled, axis=1)))

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled.argmax(axis=1)
)

# Build the model
num_classes = y_resampled.shape[1]
model = Sequential([
    Dense(256, activation="relu", kernel_regularizer=l2(0.001), input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(128, activation="relu", kernel_regularizer=l2(0.001)),
    Dropout(0.4),
    Dense(num_classes, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save the model
model.save("conference_classifier.h5")