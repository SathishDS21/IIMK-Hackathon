import openai
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import PyPDF2
from tqdm import tqdm
from joblib import dump, load
import time

input_dir = "/Users/sathishm/Documents/IITK-Input"
output_dir = "/Users/sathishm/Documents/IITK-Output"
publishable_dir = "/Users/sathishm/Documents/Publishable"
non_publishable_dir = "/Users/sathishm/Documents/Non-Publishable"
conference_data_dir = "/Users/sathishm/Documents/Conferences"

openai.api_key = "XXXXXXXXXXXXXX"

def extract_text_from_pdf(file_path):
    try:
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            text = "".join([page.extract_text() for page in pdf_reader.pages])
        return text
    except Exception:
        return ""

def generate_rationale_gpt3_5(text, conference_name):

    prompt = (
        f"This paper is being recommended for the {conference_name} conference. Based on its content, explain why this paper aligns with the themes of {conference_name}. "
        f"Focus on its contributions, novelty, and relevance:\n\n{text[:1000]}"
    )
    try:
        start_time = time.time()
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant for evaluating scientific papers."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.7,
            top_p=0.9
        )
        end_time = time.time()
        print(f"Generated rationale in {end_time - start_time:.2f} seconds")
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"Error generating rationale: {e}")
        return "Error in rationale generation"

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

model_file = os.path.join(output_dir, "Task1_model.pkl")
pca_file = os.path.join(output_dir, "Task1_pca.pkl")
scaler_file = os.path.join(output_dir, "Task1_scaler.pkl")
conference_model_file = os.path.join(output_dir, "Task2_conference_model.pkl")
conference_pca_file = os.path.join(output_dir, "Task2_conference_pca.pkl")
conference_scaler_file = os.path.join(output_dir, "Task2_conference_scaler.pkl")

conference_mapping = {
    "CVPR": 1,
    "EMNLP": 2,
    "KDD": 3,
    "NeurIPS": 4,
    "TMLR": 5
}

if os.path.exists(model_file) and os.path.exists(pca_file) and os.path.exists(scaler_file):
    retrain = input("A trained model is found. Would you like to retrain the model? (Y/N): ").strip().upper()
else:
    retrain = 'Y'

if retrain == 'Y':
    data = []
    for dir_path, label in [(publishable_dir, 1), (non_publishable_dir, 0)]:
        for filename in tqdm(os.listdir(dir_path), desc=f"Processing {dir_path} PDFs"):
            if filename.endswith(".pdf"):
                file_path = os.path.join(dir_path, filename)
                text = extract_text_from_pdf(file_path)
                data.append({'text': text, 'label': label})
    df = pd.DataFrame(data)
    X = embedding_model.encode(df['text'].tolist())
    y = df['label']
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    dump(scaler, scaler_file)
    n_components = min(100, min(X_normalized.shape))
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_normalized)
    dump(pca, pca_file)
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)
    svm = SVC(probability=True, random_state=42, kernel='linear')
    svm.fit(X_train, y_train)
    dump(svm, model_file)
else:
    model = load(model_file)
    pca = load(pca_file)
    scaler = load(scaler_file)

if os.path.exists(conference_model_file) and os.path.exists(conference_pca_file) and os.path.exists(conference_scaler_file):
    retrain_conference_model = input("A conference recommendation model is found. Would you like to retrain the model? (Y/N): ").strip().upper()
else:
    retrain_conference_model = 'Y'

if retrain_conference_model == 'Y':
    conference_data = []
    for conference, label in conference_mapping.items():
        conference_folder = os.path.join(conference_data_dir, conference)
        for filename in tqdm(os.listdir(conference_folder), desc=f"Processing {conference} PDFs"):
            if filename.endswith(".pdf"):
                file_path = os.path.join(conference_folder, filename)
                text = extract_text_from_pdf(file_path)
                conference_data.append({'text': text, 'label': label})
    conference_df = pd.DataFrame(conference_data)
    X_conference = embedding_model.encode(conference_df['text'].tolist())
    y_conference = conference_df['label']
    scaler_conference = StandardScaler()
    X_conference_normalized = scaler_conference.fit_transform(X_conference)
    dump(scaler_conference, conference_scaler_file)
    n_components_conference = min(100, min(X_conference_normalized.shape))
    pca_conference = PCA(n_components=n_components_conference)
    X_conference_pca = pca_conference.fit_transform(X_conference_normalized)
    dump(pca_conference, conference_pca_file)
    X_train_conference, X_test_conference, y_train_conference, y_test_conference = train_test_split(
        X_conference_pca, y_conference, test_size=0.3, random_state=42
    )
    svm_conference = SVC(probability=True, random_state=42, kernel='linear')
    svm_conference.fit(X_train_conference, y_train_conference)
    dump(svm_conference, conference_model_file)
else:
    svm_conference = load(conference_model_file)
    pca_conference = load(conference_pca_file)
    scaler_conference = load(conference_scaler_file)

papers = []
for filename in tqdm(os.listdir(input_dir), desc="Classifying papers"):
    if filename.endswith(".pdf"):
        file_path = os.path.join(input_dir, filename)
        text = extract_text_from_pdf(file_path)
        if text:
            X_test = embedding_model.encode([text])
            X_test_normalized = scaler.transform(X_test)
            X_test_pca = pca.transform(X_test_normalized)
            prediction = model.predict(X_test_pca)[0]
            if prediction == 1:
                X_test_conference_normalized = scaler_conference.transform(X_test)
                X_test_conference_pca = pca_conference.transform(X_test_conference_normalized)
                conference_pred = svm_conference.predict(X_test_conference_pca)[0]
                conference_name = [k for k, v in conference_mapping.items() if v == conference_pred][0]
                rationale = generate_rationale_gpt3_5(text, conference_name)
                papers.append({
                    'title': filename,
                    'publishable': 1,
                    'conference': conference_name,
                    'reason': rationale
                })
            else:
                papers.append({
                    'title': filename,
                    'publishable': 0,
                    'conference': 'NA',
                    'reason': 'NA'
                })

results_df = pd.DataFrame(papers)
results_df.to_csv(os.path.join(output_dir, "Quantrix_Squad_Stage2.csv"), index=False, encoding='utf-8')