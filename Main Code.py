import os
import time
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import PyPDF2
from tqdm import tqdm
from joblib import dump, load
from collections import Counter
import multiprocessing

os.environ["TOKENIZERS_PARALLELISM"] = "false"
multiprocessing.set_start_method('spawn', force=True)

input_dir = "/Users/sathishm/Documents/IITK-Input"
output_dir = "/Users/sathishm/Documents/IITK-Output"
publishable_dir = "/Users/sathishm/Documents/Publishable"
non_publishable_dir = "/Users/sathishm/Documents/Non-Publishable"

model_file = os.path.join(output_dir, "Task1_model.pkl")
pca_file = os.path.join(output_dir, "Task1_pca.pkl")

def extract_text_from_pdf(file_path):
    try:
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
        return text
    except Exception:
        return ""

if os.path.exists(model_file) and os.path.exists(pca_file):
    retrain = input("A trained model is found. Would you like to retrain the model? (Y/N): ").strip().upper()
else:
    print("No trained model found. Proceeding to train the model...")
    retrain = 'Y'

if retrain == 'Y':
    data = []
    print("Processing PDFs...")
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
    print(f"Class distribution: {Counter(df['label'])}")

    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    start_time = time.time()
    X = embedding_model.encode(df['text'].tolist())
    y = df['label']
    vectorization_latency = time.time() - start_time

    max_components = min(X.shape[0], X.shape[1])
    n_components = min(100, max_components)
    print(f"Using PCA with n_components={n_components}")
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    pca_latency = time.time() - start_time

    dump(pca, pca_file)

    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

    print("Tuning model...")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }
    rf = RandomForestClassifier(class_weight='balanced', random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='f1', verbose=0, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    model = grid_search.best_estimator_

    dump(model, model_file)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_pca, y, cv=cv, scoring='f1')

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\nModel Scores:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"Cross-Validation F1 Score (Mean): {cv_scores.mean():.2f}")
else:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    model = load(model_file)
    pca = load(pca_file)
    print("Loaded the existing model and PCA.")

    print("Evaluating the loaded model...")
    df = pd.DataFrame({'text': ["Dummy data"] * 10, 'label': [0] * 5 + [1] * 5})
    X = embedding_model.encode(df['text'].tolist())
    X_pca = pca.transform(X)
    y = df['label']

    y_pred = model.predict(X_pca)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    print("\nLoaded Model Evaluation Scores:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

papers = []
print("Classifying input PDFs...")
start_time = time.time()
for filename in tqdm(os.listdir(input_dir), desc="Classifying input PDFs"):
    if filename.endswith(".pdf"):
        file_path = os.path.join(input_dir, filename)
        text = extract_text_from_pdf(file_path)
        if text:
            X_test = embedding_model.encode([text])
            X_test_pca = pca.transform(X_test)
            prediction = model.predict(X_test_pca)[0]
            papers.append({'title': filename, 'publishable': prediction})
classification_latency = time.time() - start_time

results_file_path = os.path.join(output_dir, "paper_classification_results.csv")
results_df = pd.DataFrame(papers)
results_df.to_csv(results_file_path, index=False)

if retrain == 'Y':
    latency_data = {
        "Step": ["Publishable PDFs", "Non-Publishable PDFs", "Vectorization", "PCA Dimensionality Reduction",
                 "Model Training and Tuning", "Classification"],
        "Latency (seconds)": [
            locals().get('publishable_latency', 'N/A'),
            locals().get('non_publishable_latency', 'N/A'),
            locals().get('vectorization_latency', 'N/A'),
            locals().get('pca_latency', 'N/A'),
            locals().get('training_latency', 'N/A'),
            classification_latency
        ]
    }
else:
    latency_data = {
        "Step": ["Classification"],
        "Latency (seconds)": [classification_latency]
    }

latency_df = pd.DataFrame(latency_data)
print("\nLatency Summary:")
print(latency_df)