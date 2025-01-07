import os
import PyPDF2
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

def extract_text_from_pdf(file_path):
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = " ".join(page.extract_text() for page in reader.pages if page.extract_text())
        return text
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""


def load_data(directory, label):
    texts = []
    labels = []
    file_names = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.pdf'):
                file_path = os.path.join(root, file)
                text = extract_text_from_pdf(file_path)
                if text:
                    texts.append(text)
                    labels.append(label)
                    file_names.append(file)
    return texts, labels, file_names


publishable_dir = "/Users/sathishm/Downloads/Publishable"
non_publishable_dir = "/Users/sathishm/Downloads/Non-Publishable"

publishable_texts, publishable_labels, _ = load_data(publishable_dir, 1)
non_publishable_texts, non_publishable_labels, _ = load_data(non_publishable_dir, 0)

texts = publishable_texts + non_publishable_texts
labels = publishable_labels + non_publishable_labels

vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(texts).toarray()
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

joblib.dump(model, "publishability_classifier.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

input_dir = "/Users/sathishm/Downloads/IITK-Input"
output_dir = "/Users/sathishm/Downloads/IITK"
os.makedirs(output_dir, exist_ok=True)
output_file_path = os.path.join(output_dir, "publishability_results.xlsx")

input_texts, _, input_files = load_data(input_dir, None)

results = []
for file_name, text in zip(input_files, input_texts):
    if text:  # Process only if text is extracted
        features = vectorizer.transform([text]).toarray()
        prediction = model.predict(features)
        results.append({"Paper ID": file_name, "Publishable": int(prediction[0])})

df = pd.DataFrame(results)
df.to_excel(output_file_path, index=False)

print(f"Assessment results saved to {output_file_path}")