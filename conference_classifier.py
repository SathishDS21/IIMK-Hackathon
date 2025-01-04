import os
import fitz
import pdfplumber
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from transformers import pipeline
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
import shutil

# Initialize FastAPI app
app = FastAPI()

# Load summarization model (Specify GPU usage)
summarizer = pipeline("summarization", model="t5-small", framework="pt", device=0)

# Set random seed
SEED = 42
np.random.seed(SEED)

# Function to extract text from PDFs
def extract_text_from_pdf(file_path):
    try:
        with fitz.open(file_path) as pdf:
            text = ""
            for page in pdf:
                text += page.get_text()
            if text.strip():
                return text.strip()
    except Exception as e:
        print(f"PyMuPDF failed for {file_path}: {e}")
    try:
        with pdfplumber.open(file_path) as pdf:
            text = " ".join(page.extract_text() for page in pdf.pages if page.extract_text())
            return text.strip()
    except Exception as e:
        print(f"pdfplumber failed for {file_path}: {e}")
        return ""

# Generate rationale for predictions
def generate_rationale(paper_content, prediction):
    try:
        truncated_content = paper_content[:1000]
        summarized_content = summarizer(truncated_content, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
        rationale = f"{summarized_content} This paper aligns with the themes of '{prediction}' due to its focus on topics and methods commonly explored in {prediction} research."
        return rationale
    except Exception as e:
        print(f"Error generating rationale: {e}")
        return f"Rationale generation failed due to an error: {str(e)}"

# Load dummy training data (replace this with actual Publishable and Non-Publishable PDFs)
texts = ["This is a publishable research paper on AI.", "This paper does not meet publishable standards."]
labels = ["Publishable", "Non-Publishable"]

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Vectorize text
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(texts).toarray()
y = to_categorical(labels_encoded, num_classes=len(label_encoder.classes_))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

# Build the model
model = Sequential([
    Dense(128, activation='relu', kernel_regularizer=l2(0.001), input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.4),
    Dense(len(label_encoder.classes_), activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Handle insufficient data for validation
if len(X_train) < 5:
    print("Insufficient data for validation split. Skipping validation.")
    model.fit(X_train, y_train, epochs=5, batch_size=16)
else:
    model.fit(X_train, y_train, epochs=5, batch_size=16, validation_split=0.2)

# Define conferences for recommendations
conferences = ["CVPR", "NeurIPS", "DAA", "EMNLP", "KDD"]

# Web interface
@app.get("/")
def upload_form():
    return HTMLResponse(content="""
    <html>
        <head>
            <title>Research Paper Classification</title>
        </head>
        <body>
            <h2>Research Paper Classification</h2>
            <form action="/upload/" enctype="multipart/form-data" method="post">
                <label for="files">Please upload the paper:</label><br><br>
                <input name="files" type="file" multiple><br><br>
                <button type="submit">Submit</button>
                <button type="reset">Clear</button>
            </form>
        </body>
    </html>
    """)

@app.post("/upload/")
async def classify_files(files: list[UploadFile] = File(...)):
    input_texts = []
    input_files = []
    for file in files:
        file_path = f"/tmp/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        text = extract_text_from_pdf(file_path)
        if text:
            input_texts.append(text)
            input_files.append(file.filename)

    if not input_texts:
        return HTMLResponse(content="<h2>No valid PDF files were uploaded.</h2>")

    # Vectorize and predict
    input_vectors = vectorizer.transform(input_texts).toarray()
    predictions = model.predict(input_vectors)
    predicted_labels = label_encoder.inverse_transform(np.argmax(predictions, axis=1))

    # Generate results
    results = []
    for file, text, prediction in zip(input_files, input_texts, predicted_labels):
        # Randomly assign a conference for demonstration purposes
        conference = np.random.choice(conferences)
        rationale = generate_rationale(text, conference)
        results.append({
            "File Name": file,
            "Prediction": prediction,
            "Conference": conference,
            "Rationale": rationale
        })

    # Create HTML table
    table_rows = "".join(
        f"<tr><td>{result['File Name']}</td><td>{result['Prediction']}</td><td>{result['Conference']}</td><td>{result['Rationale']}</td></tr>"
        for result in results
    )
    html_content = f"""
    <html>
        <body>
            <h2>Classification Results</h2>
            <table border="1">
                <tr><th>File Name</th><th>Prediction</th><th>Conference</th><th>Rationale</th></tr>
                {table_rows}
            </table>
            <br>
            <form action="/" method="get">
                <button type="submit">Clear</button>
            </form>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)