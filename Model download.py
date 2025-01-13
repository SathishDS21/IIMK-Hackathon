import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from joblib import load
from tqdm import tqdm
from PyPDF2 import PdfReader
import pathway as pw

# Google Drive Credentials and File IDs
service_user_credentials_file = "/Users/sathishm/Documents/IITK-Input copy/quantrixsquad-e3e129adda77.json"
task1_model_id = "14vrqhvpIpvZKJZtoszqGHrBshR3W-2Gs"
task1_pca_id = "1BuQsaxWnfk-wSLrNeGgJTe5aZddPpSmf"
task1_scaler_id = "1Zh5tRkOJrFPSvUlpyneVTdho6OH-8dZU"
task2_model_id = "1WeBoLK0T3oehZMRjZR_Sfn8cWweKUo2M"
task2_pca_id = "1GvdTlfzpYRLcVbp9Omq5vwHbRFlKtM0J"
task2_scaler_id = "1NkrncKeTbAg9Z_GbZo8GTCuSKiW0JnMQ"
input_dir_id = "1lgzaa_odpCQQ0C4SfFNV7TD1j6wIzDSB"

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Helper function to extract text from PDFs
def extract_text_from_pdf(content):
    try:
        with open("/tmp/temp_pdf.pdf", "wb") as temp_pdf:
            temp_pdf.write(content)
        reader = PdfReader("/tmp/temp_pdf.pdf")
        text = "".join(page.extract_text() for page in reader.pages)
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

# Function to process models from Google Drive
def load_model_from_gdrive(object_id, service_user_credentials_file, filename):
    """Download and save a model file locally from Google Drive."""
    table = pw.io.gdrive.read(object_id=object_id, service_user_credentials_file=service_user_credentials_file)
    results = pw.run(table)  # Execute the computation graph to retrieve results
    for row in results:
        content = row["content"]
        file_path = os.path.join("/tmp", filename)
        with open(file_path, "wb") as f:
            f.write(content)  # Save the file content as binary
        return file_path
    return None

# Load Task 1 and Task 2 models
task1_model_path = load_model_from_gdrive(task1_model_id, service_user_credentials_file, "task1_model.pkl")
task1_pca_path = load_model_from_gdrive(task1_pca_id, service_user_credentials_file, "task1_pca.pkl")
task1_scaler_path = load_model_from_gdrive(task1_scaler_id, service_user_credentials_file, "task1_scaler.pkl")
task2_model_path = load_model_from_gdrive(task2_model_id, service_user_credentials_file, "task2_model.pkl")
task2_pca_path = load_model_from_gdrive(task2_pca_id, service_user_credentials_file, "task2_pca.pkl")
task2_scaler_path = load_model_from_gdrive(task2_scaler_id, service_user_credentials_file, "task2_scaler.pkl")

# Load models into memory
task1_model = load(task1_model_path)
task1_pca = load(task1_pca_path)
task1_scaler = load(task1_scaler_path)
task2_model = load(task2_model_path)
task2_pca = load(task2_pca_path)
task2_scaler = load(task2_scaler_path)

# Process input papers
input_table = pw.io.gdrive.read(object_id=input_dir_id, service_user_credentials_file=service_user_credentials_file)
input_results = pw.run(input_table)  # Execute the computation graph

results = []
for row in input_results:
    filename = row["filename"]
    content = row["content"]
    text = extract_text_from_pdf(content)
    if not text:
        results.append({"filename": filename, "publishable": 0, "conference": "NA", "reason": "Empty content"})
        continue

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
        conference_name = next((k for k, v in conference_mapping.items() if v == conference_pred), "Unknown")

        rationale = f"Recommended for {conference_name}"  # Replace with GPT rationale if needed
        results.append({"filename": filename, "publishable": 1, "conference": conference_name, "reason": rationale})
    else:
        results.append({"filename": filename, "publishable": 0, "conference": "NA", "reason": "Not publishable"})

# Save results
output_dir = "/path/to/local/output"
os.makedirs(output_dir, exist_ok=True)
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(output_dir, "paper_classification_results.csv"), index=False)

print("Processing complete. Results saved to:", os.path.join(output_dir, "paper_classification_results.csv"))