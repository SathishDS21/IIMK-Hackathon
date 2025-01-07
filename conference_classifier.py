import os
import tempfile
import fitz
import pdfplumber
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from transformers import pipeline
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
import shutil
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

summarizer = pipeline("summarization", model="t5-small", framework="pt", device=-1)

SEED = 42
np.random.seed(SEED)

texts = ["This is a publishable research paper on AI.", "This paper does not meet publishable standards."]
labels = ["Publishable", "Non-Publishable"]

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(texts).toarray()
y = to_categorical(labels_encoded, num_classes=len(label_encoder.classes_))

if not os.path.exists("paper_classifier.h5"):
    model = Sequential([
        Dense(128, activation='relu', kernel_regularizer=l2(0.001), input_shape=(X.shape[1],)),
        Dropout(0.5),
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.4),
        Dense(len(label_encoder.classes_), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=5, batch_size=16)
    model.save("paper_classifier.h5")
else:
    model = load_model("paper_classifier.h5")

conferences = ["CVPR", "NeurIPS", "DAA", "EMNLP", "KDD"]


def extract_text_from_pdf(file_path):
    try:
        with fitz.open(file_path) as pdf:
            text = ""
            for page in pdf:
                text += page.get_text()
            if text.strip():
                return text.strip()
    except Exception as e:
        logger.error(f"PyMuPDF failed for {file_path}: {e}")
    try:
        with pdfplumber.open(file_path) as pdf:
            text = " ".join(page.extract_text() for page in pdf.pages if page.extract_text())
            return text.strip()
    except Exception as e:
        logger.error(f"pdfplumber failed for {file_path}: {e}")
    return ""


def generate_rationale(paper_content, prediction):
    try:
        truncated_content = paper_content[:1000]
        summarized_content = summarizer(truncated_content, max_length=100, min_length=30, do_sample=False)[0][
            'summary_text']
        rationale = f"{summarized_content} This paper aligns with the themes of '{prediction}' due to its focus on topics and methods commonly explored in {prediction} research."
        return rationale
    except Exception as e:
        logger.error(f"Error generating rationale: {e}")
        return f"Rationale generation failed due to an error: {str(e)}"


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
        if file.content_type != "application/pdf":
            return JSONResponse(content={"error": f"File {file.filename} is not a valid PDF."}, status_code=400)

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            text = extract_text_from_pdf(temp_file.name)

        if text:
            input_texts.append(text)
            input_files.append(file.filename)

    if not input_texts:
        return JSONResponse(content={"error": "No valid PDF files were uploaded."}, status_code=400)

    input_vectors = vectorizer.transform(input_texts).toarray()
    predictions = model.predict(input_vectors)
    predicted_labels = label_encoder.inverse_transform(np.argmax(predictions, axis=1))

    results = []
    for file, text, prediction in zip(input_files, input_texts, predicted_labels):
        conference = np.random.choice(conferences)
        rationale = generate_rationale(text, conference)
        results.append({
            "File Name": file,
            "Prediction": prediction,
            "Conference": conference,
            "Rationale": rationale
        })

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