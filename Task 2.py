import os
import time
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import PyPDF2
from tqdm import tqdm
from joblib import dump, load
from collections import Counter
import multiprocessing
from llm_app import pathway_connector

os.environ["TOKENIZERS_PARALLELISM"] = "false"
multiprocessing.set_start_method('spawn', force=True)

# Directories
input_dir = "/Users/sathishm/Documents/IITK-Input"
output_dir = "/Users/sathishm/Documents/IITK-Output"
publishable_dir = "/Users/sathishm/Documents/Publishable"
non_publishable_dir = "/Users/sathishm/Documents/Non-Publishable"
conference_dirs = {
    "CVPR": "/Users/sathishm/Downloads/CVPR",
    "EMNLP": "/Users/sathishm/Downloads/EMNLP",
    "KDD": "/Users/sathishm/Downloads/KDD",
    "NeurIPS": "/Users/sathishm/Downloads/NeurIPS",
    "TMLR": "/Users/sathishm/Downloads/TMLR"
}

# Model and files
model_file = os.path.join(output_dir, "Task1_model.pkl")
pca_file = os.path.join(output_dir, "Task1_pca.pkl")
scaler_file = os.path.join(output_dir, "Task1_scaler.pkl")
conference_model_file = os.path.join(output_dir, "Task2_conference_model.pkl")
conference_pca_file = os.path.join(output_dir, "Task2_conference_pca.pkl")
conference_scaler_file = os.path.join(output_dir, "Task2_conference_scaler.pkl")

# Pathway Connectors Initialization
connector = pathway_connector.PathwayConnector(api_key="2278DD-9E5137-37FB24-70D11F-C63F96-V3")  # Replace with your API key

# Extract text from PDF
def extract_text_from_pdf(file_path):
    try:
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num].extract_text()
        return text
    except Exception:
        return ""

# Load or train the Task 1 model
if os.path.exists(model_file) and os.path.exists(pca_file) and os.path.exists(scaler_file):
    retrain = input("A trained model is found. Would you like to retrain the model? (Y/N): ").strip().upper()
else:
    print("No trained model found. Proceeding to train the model...")
    retrain = 'Y'

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

if retrain == 'Y':
    data = []
    for label, directory in [(1, publishable_dir), (0, non_publishable_dir)]:
        for filename in tqdm(os.listdir(directory), desc=f"Processing PDFs in {directory}"):
            if filename.endswith(".pdf"):
                file_path = os.path.join(directory, filename)
                text = extract_text_from_pdf(file_path)
                data.append({'text': text, 'label': label})
    df = pd.DataFrame(data)
    X = embedding_model.encode(df['text'].tolist())
    y = df['label']

    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    dump(scaler, scaler_file)

    pca = PCA(n_components=min(100, X_normalized.shape[1]))
    X_pca = pca.fit_transform(X_normalized)
    dump(pca, pca_file)

    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)
    svm = SVC(probability=True, random_state=42)
    param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']}
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    model = grid_search.best_estimator_
    dump(model, model_file)

else:
    model = load(model_file)
    pca = load(pca_file)
    scaler = load(scaler_file)

# Task 2: Train or Load Conference Recommendation Model
if retrain == 'Y':
    conference_data = []
    for conference, directory in conference_dirs.items():
        for filename in tqdm(os.listdir(directory), desc=f"Processing {conference} PDFs"):
            if filename.endswith(".pdf"):
                file_path = os.path.join(directory, filename)
                text = extract_text_from_pdf(file_path)
                conference_data.append({'text': text, 'label': conference})

    conference_df = pd.DataFrame(conference_data)
    conference_df['label'] = conference_df['label'].astype('category').cat.codes
    label_mapping = dict(enumerate(conference_df['label'].astype('category').cat.categories))

    X_conference = embedding_model.encode(conference_df['text'].tolist())
    y_conference = conference_df['label']

    conference_scaler = StandardScaler()
    X_conference_normalized = conference_scaler.fit_transform(X_conference)
    dump(conference_scaler, conference_scaler_file)

    conference_pca = PCA(n_components=min(100, X_conference_normalized.shape[1]))
    X_conference_pca = conference_pca.fit_transform(X_conference_normalized)
    dump(conference_pca, conference_pca_file)

    X_train, X_test, y_train, y_test = train_test_split(X_conference_pca, y_conference, test_size=0.3, random_state=42)
    grid_search.fit(X_train, y_train)
    conference_model = grid_search.best_estimator_
    dump(conference_model, conference_model_file)

else:
    conference_model = load(conference_model_file)
    conference_pca = load(conference_pca_file)
    conference_scaler = load(conference_scaler_file)

# Classify Papers
papers = []
for filename in tqdm(os.listdir(input_dir), desc="Classifying input PDFs"):
    if filename.endswith(".pdf"):
        file_path = os.path.join(input_dir, filename)
        text = extract_text_from_pdf(file_path)
        X_test = embedding_model.encode([text])
        X_test_normalized = scaler.transform(X_test)
        X_test_pca = pca.transform(X_test_normalized)
        prediction = model.predict(X_test_pca)[0]
        paper = {'title': filename, 'publishable': prediction}
        if prediction == 1:
            X_conference_test_normalized = conference_scaler.transform(X_test)
            X_conference_test_pca = conference_pca.transform(X_conference_test_normalized)
            conference_prediction = conference_model.predict(X_conference_test_pca)[0]
            paper['conference'] = label_mapping[conference_prediction]
        else:
            paper['conference'] = "NA"
        papers.append(paper)

# Task 3: GenAI Recommendations
for paper in papers:
    if paper['publishable'] == 1:
        conference = paper['conference']
        try:
            response = connector.generate_reason(
                prompt=f"Explain why this paper is suitable for the {conference} conference: {paper['title']}"
            )
            paper['recommendation_reason'] = response
        except Exception as e:
            paper['recommendation_reason'] = f"Error generating reason: {str(e)}"
    else:
        paper['recommendation_reason'] = "NA"

# Save Final Results
results_df = pd.DataFrame(papers)
results_df.to_csv(os.path.join(output_dir, "final_paper_classification_results.csv"), index=False)

print("Processing complete! Results saved.")