from flask import Flask, render_template, request, send_file
from flask_bootstrap import Bootstrap
import sqlite3
from werkzeug.utils import secure_filename
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import PyPDF2
import docx
import spacy
import re

app = Flask(__name__)
Bootstrap(app)

def extract_text_from_file(filepath):
    _, extension = os.path.splitext(filepath)
    if extension.lower() == '.pdf':
        with open(filepath, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
    elif extension.lower() == '.docx':
        doc = docx.Document(filepath)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    else:
        raise ValueError("Unsupported file type")
    return text

def analyze_sentiment(text):
    max_length = sentiment_tokenizer.model_max_length
    chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
    
    results = {"positive": 0, "negative": 0, "neutral": 0}
    
    for chunk in chunks:
        inputs = sentiment_tokenizer(chunk, return_tensors="pt", truncation=True, max_length=max_length)
        outputs = sentiment_model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        results["positive"] += predictions[0][0].item()
        results["negative"] += predictions[0][1].item()
        results["neutral"] += predictions[0][2].item()
        
    total = sum(results.values())
    for key in results:
        results[key] /= total
        
    return results

def extract_entities(text):
    doc = ner_model(text)
    entities = {}
    for ent in doc.ents:
        if ent.label_ not in entities:
            entities[ent.label_] = []
        entities[ent.label_].append(ent.text)
    return entities

def extract_financial_metrics(text):
    metrics = {}
    patterns = {
        'revenue': r'(\$\s?\d+(?:,\d+)*(?:\.\d+)?\s*(?:billion|million|thousand)?\s*(?:in revenue|revenue))',
        'profit': r'(\$\s?\d+(?:,\d+)*(?:\.\d+)?\s*(?:billion|million|thousand)?\s*(?:in (?:net )?profit|(?:net )?profit))',
        'assets': r'(\$\s?\d+(?:,\d+)*(?:\.\d+)?\s*(?:billion|million|thousand)?\s*(?:in (?:total )?assets|(?:total )?assets))',
    }
    
    for metric, pattern in patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            # Clean the matches to remove leading/trailing whitespace and newlines
            cleaned_matches = [match.strip() for match in matches]
            metrics[metric] = cleaned_matches
    return metrics

def analyze_esg(text):
    max_length = 512
    chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
    
    results = {}
    for chunk in chunks:
        esg_scores = esg_classifier(chunk)
        for score in esg_scores:
            label = score['label']
            if label not in results:
                results[label] = 0
            results[label] += score['score']
            
    # Normalize results
    total = sum(results.values())
    for key in results:
        results[key] /= total
        
    return results

def generate_summary(sentiment_result, entities, financial_metrics, esg_result):
    sentiment = max(sentiment_result, key=sentiment_result.get)
    confidence = sentiment_result[sentiment]
    
    financial_highlights = ""
    if financial_metrics:
        for key, values in financial_metrics.items():
            financial_highlights += f"{key.capitalize()}: {', '.join(values)}\n"
    else:
        financial_highlights = "No financial metrics extracted."
    
    summary = {
        "executive_summary": f"The overall sentiment of this report is {sentiment} with {confidence:.2%} confidence.",
        "financial_highlights": financial_highlights.strip(),
        "business_highlights": "Key entities mentioned:\n" + '\n'.join([f"{k}: {', '.join(v[:5])}" for k, v in entities.items()]),
        "risk_analysis": "A detailed risk analysis should be conducted by financial experts.",
        "peer_comparison": "Comparative analysis with peers requires additional data and expert interpretation.",
        "outlook": f"Based on the {sentiment} sentiment, the outlook might be {'positive' if sentiment == 'positive' else 'challenging' if sentiment == 'negative' else 'stable'}. However, this should be verified with detailed financial analysis.",
        "esg_analysis": "ESG Analysis:\n" + '\n'.join([f"{k}: {v:.2%}" for k, v in esg_result.items()])
    }
    return summary


def process_document(filepath):
    text = extract_text_from_file(filepath)
    
    sentiment_result = analyze_sentiment(text)
    entities = extract_entities(text)
    financial_metrics = extract_financial_metrics(text)
    esg_result = analyze_esg(text)
    
    summary = generate_summary(sentiment_result, entities, financial_metrics, esg_result)
    return summary

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load models
sentiment_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
sentiment_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
ner_model = spacy.load("en_core_web_sm")
esg_classifier = pipeline("text-classification", model="yiyanghkust/finbert-esg")

# SQLite database setup
def init_db():
    conn = sqlite3.connect('reports.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS reports
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                summary TEXT)''')
    conn.commit()
    conn.close()
    
init_db()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the file with our NLP pipeline
            summary = process_document(filepath)
            
            # Store in SQLite
            conn = sqlite3.connect('reports.db')
            c = conn.cursor()
            c.execute("INSERT INTO reports (filename, summary) VALUES (?, ?)", (filename, str(summary)))
            conn.commit()
            report_id = c.lastrowid
            conn.close()
            
            return render_template('summary.html', summary=summary, report_id=report_id)
    return render_template('upload.html')

@app.route('/download_pdf/<int:report_id>')
def download_pdf(report_id):
    conn = sqlite3.connect('reports.db')
    c = conn.cursor()
    c.execute("SELECT filename FROM reports WHERE id = ?", (report_id,))
    result = c.fetchone()
    conn.close()
    
    if result:
        filename = result[0]
        return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), as_attachment=True)
    else:
        return "File not found", 404

if __name__ == '__main__':
    app.run(debug=True)
