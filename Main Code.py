import os
import time
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import RandomOverSampler
import PyPDF2
from tqdm import tqdm

input_dir = "/Users/sathishm/Documents/IITK-Input"
output_dir = "/Users/sathishm/Documents/IITK-Output"
publishable_dir = "/Users/sathishm/Documents/Publishable"
non_publishable_dir = "/Users/sathishm/Documents/Non-Publishable"

def extract_text_from_pdf(file_path):
    try:
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
        return text
    except Exception as e:
        return ""

data = []
start_time = time.time()
for filename in tqdm(os.listdir(publishable_dir), desc="Processing publishable PDFs"):
    if filename.endswith(".pdf"):
        file_path = os.path.join(publishable_dir, filename)
        text = extract_text_from_pdf(file_path)
        data.append({'text': text, 'label': 1})
publishable_latency = time.time() - start_time

start_time = time.time()
for filename in tqdm(os.listdir(non_publishable_dir), desc="Processing non-publishable PDFs"):
    if filename.endswith(".pdf"):
        file_path = os.path.join(non_publishable_dir, filename)
        text = extract_text_from_pdf(file_path)
        data.append({'text': text, 'label': 0})
non_publishable_latency = time.time() - start_time

df = pd.DataFrame(data)

start_time = time.time()
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(df['text'])
y = df['label']
vectorization_latency = time.time() - start_time

start_time = time.time()
resampler = RandomOverSampler(random_state=42)
X_resampled, y_resampled = resampler.fit_resample(X, y)
oversampling_latency = time.time() - start_time

start_time = time.time()
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_resampled, y_resampled)
training_latency = time.time() - start_time

loo = LeaveOneOut()
predictions = []
true_labels = []
loo_start_time = time.time()

for train_index, test_index in tqdm(loo.split(X_resampled), desc="LOO Cross-Validation"):
    X_train, X_test = X_resampled[train_index], X_resampled[test_index]
    y_train, y_test = y_resampled[train_index], y_resampled[test_index]

    if len(set(y_train)) < 2:
        continue

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions.extend(y_pred)
    true_labels.extend(y_test)

loo_latency = time.time() - loo_start_time

if predictions and true_labels:
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)

    metrics_data = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "Value": [accuracy, precision, recall, f1]
    }
    metrics_df = pd.DataFrame(metrics_data)
    print(metrics_df)

papers = []
start_time = time.time()
for filename in tqdm(os.listdir(input_dir), desc="Classifying input PDFs"):
    if filename.endswith(".pdf"):
        file_path = os.path.join(input_dir, filename)
        text = extract_text_from_pdf(file_path)
        if text:
            X_test = vectorizer.transform([text])
            prediction = model.predict(X_test)[0]
            papers.append({'title': filename, 'publishable': prediction})
classification_latency = time.time() - start_time

results_file_path = os.path.join(output_dir, "paper_classification_results_resampled.csv")
results_df = pd.DataFrame(papers)
results_df.to_csv(results_file_path, index=False)

latency_data = {
    "Step": ["Publishable PDFs", "Non-Publishable PDFs", "Vectorization", "Oversampling",
             "Model Training", "LOO Cross-Validation", "Classification"],
    "Latency (seconds)": [publishable_latency, non_publishable_latency, vectorization_latency,
                          oversampling_latency, training_latency, loo_latency, classification_latency]
}
latency_df = pd.DataFrame(latency_data)
print("\nLatency Summary:")
print(latency_df)