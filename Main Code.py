import openai
import os
import pandas as pd
import pathway as pw
from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from joblib import dump, load
from tqdm import tqdm
import time
from PyPDF2 import PdfReader

# OpenAI API key
openai.api_key = "sk-your-api-key"

# Pathway credentials and object IDs
service_user_credentials_file = "/Users/sathishm/Documents/IITK-Input copy/quantrixsquad-e3e129adda77.json"
task1_model_id = "GOOGLE_DRIVE_FILE_ID_FOR_TASK1_MODEL"
task1_pca_id = "GOOGLE_DRIVE_FILE_ID_FOR_TASK1_PCA"
task1_scaler_id = "GOOGLE_DRIVE_FILE_ID_FOR_TASK1_SCALER"
task2_model_id = "GOOGLE_DRIVE_FILE_ID_FOR_TASK2_MODEL"
task2_pca_id = "GOOGLE_DRIVE_FILE_ID_FOR_TASK2_PCA"
task2_scaler_id = "GOOGLE_DRIVE_FILE_ID_FOR_TASK2_SCALER"
input_dir_id = "GOOGLE_DRIVE_FOLDER_ID_FOR_INPUT_PAPERS"

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Helper function to process PDFs and extract text
def extract_text_from_pdf(content):
    """Extracts text content from a PDF binary."""
    try:
        reader = PdfReader(content)
        text = "".join(page.extract_text() for page in reader.pages)
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

# Load models from Google Drive using Pathway
@pw.transform
def load_models(row):
    """Save model files locally from Google Drive content."""
    content = row.get("content", b"")
    filename = row.get("filename", "model.pkl")
    folder = "/path/to/local/models"
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, filename)
    with open(file_path, "wb") as f:
        f.write(content)
    return {"file_path": file_path}

# Task 1 model loading
task1_model_table = pw.io.gdrive.read(
    object_id=task1_model_id, service_user_credentials_file=service_user_credentials_file
)
task1_pca_table = pw.io.gdrive.read(
    object_id=task1_pca_id, service_user_credentials_file=service_user_credentials_file
)
task1_scaler_table = pw.io.gdrive.read(
    object_id=task1_scaler_id, service_user_credentials_file=service_user_credentials_file
)

task1_model_result = load_models(task1_model_table)
task1_pca_result = load_models(task1_pca_table)
task1_scaler_result = load_models(task1_scaler_table)

# Task 2 model loading
task2_model_table = pw.io.gdrive.read(
    object_id=task2_model_id, service_user_credentials_file=service_user_credentials_file
)
task2_pca_table = pw.io.gdrive.read(
    object_id=task2_pca_id, service_user_credentials_file=service_user_credentials_file
)
task2_scaler_table = pw.io.gdrive.read(
    object_id=task2_scaler_id, service_user_credentials_file=service_user_credentials_file
)

task2_model_result = load_models(task2_model_table)
task2_pca_result = load_models(task2_pca_table)
task2_scaler_result = load_models(task2_scaler_table)

# Run Pathway to load models
task1_model_path = pw.run(task1_model_result)
task1_pca_path = pw.run(task1_pca_result)
task1_scaler_path = pw.run(task1_scaler_result)
task2_model_path = pw.run(task2_model_result)
task2_pca_path = pw.run(task2_pca_result)
task2_scaler_path = pw.run(task2_scaler_result)

# Load models into memory
task1_model = load(task1_model_path[0]['file_path'])
task1_pca = load(task1_pca_path[0]['file_path'])
task1_scaler = load(task1_scaler_path[0]['file_path'])
task2_model = load(task2_model_path[0]['file_path'])
task2_pca = load(task2_pca_path[0]['file_path'])
task2_scaler = load(task2_scaler_path[0]['file_path'])

# Load input papers from Google Drive
input_table = pw.io.gdrive.read(
    object_id=input_dir_id, service_user_credentials_file=service_user_credentials_file
)

@pw.transform
def process_papers(row):
    """Process each input paper, classify, and recommend."""
    filename = row.get("filename", "Unknown")
    content = row.get("content", b"")
    try:
        text = extract_text_from_pdf(content)
        if not text:
            return {"filename": filename, "publishable": 0, "conference": "NA", "reason": "NA"}

        # Task 1: Publishability classification
        X_test = embedding_model.encode([text])
        X_test_normalized = task1_scaler.transform(X_test)
        X_test_pca = task1_pca.transform(X_test_normalized)
        prediction = task1_model.predict(X_test_pca)[0]

        if prediction == 1:
            # Task 2: Conference recommendation
            X_test_conference_normalized = task2_scaler.transform(X_test)
            X_test_conference_pca = task2_pca.transform(X_test_conference_normalized)
            conference_pred = task2_model.predict(X_test_conference_pca)[0]
            conference_name = [k for k, v in conference_mapping.items() if v == conference_pred][0]

            # Generate rationale
            rationale = generate_rationale_gpt3_5(text, conference_name)

            return {"filename": filename, "publishable": 1, "conference": conference_name, "reason": rationale}
        else:
            return {"filename": filename, "publishable": 0, "conference": "NA", "reason": "NA"}
    except Exception as e:
        return {"filename": filename, "publishable": 0, "conference": "NA", "reason": f"Error: {e}"}

processed_papers = process_papers(input_table)
results = pw.run(processed_papers)

# Save results
output_dir = "/path/to/local/output"
os.makedirs(output_dir, exist_ok=True)
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(output_dir, "paper_classification_results.csv"), index=False)