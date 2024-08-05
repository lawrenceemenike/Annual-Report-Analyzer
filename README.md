# Annual Report Analyzer

Welcome to the Annual Report Analyzer! I decided to embark on this project when my submission to an LLM course by WandB was rejected. The simple app is designed to help you upload and analyze annual reports in PDF or DOCX format, extracting key insights such as sentiment analysis, named entities, financial metrics, and ESG (Environmental, Social, and Governance) scores.

## Features

- **Sentiment Analysis**: Determines the overall sentiment of the report.
- **Named Entity Recognition**: Extracts and categorizes key entities mentioned in the report.
- **Financial Metrics Extraction**: Identifies and extracts key financial metrics such as revenue, profit, and assets.
- **ESG Analysis**: Analyzes the report based on ESG criteria.

## Requirements

To get started, you'll need to install the necessary Python packages. These are listed in the `requirements.txt` file.

## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/yourusername/annual-report-analyzer.git
    cd annual-report-analyzer
    ```

2. **Create a virtual environment** (optional but recommended):
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Download spaCy model**:
    ```sh
    python -m spacy download en_core_web_sm
    ```

## Usage

1. **Run the Flask app**:
    ```sh
    python main.py
    ```

2. **Upload your annual report**:
    - Navigate to `http://127.0.0.1:5000/` in your web browser.
    - Use the upload form to submit your PDF or DOCX report.

3. **View the analysis summary**:
    - After uploading, the app will process the document and display a summary page with the extracted insights.
    - You can download the original report using the provided link.

## File Structure

- `main.py`: The main application script.
- `templates/`: Directory containing HTML templates (`upload.html` and `summary.html`).
- `static/`: Directory for static files like CSS.
- `requirements.txt`: List of required Python packages.
- `README.md`: This file.

## Technologies Used

- **Flask**: Web framework for Python.
- **Flask-Bootstrap**: For integrating Bootstrap with Flask.
- **transformers**: For sentiment analysis and ESG scoring using pre-trained models.
- **PyPDF2**: For extracting text from PDF files.
- **python-docx**: For extracting text from DOCX files.
- **spaCy**: For named entity recognition.
- **SQLite**: For storing analysis summaries.

## Future Enhancements

- **Detailed Risk Analysis**: Implementing a more comprehensive risk analysis feature.
- **Peer Comparison**: Adding functionality for comparative analysis with peer reports.
- **Extended Financial Metrics**: Extracting additional financial metrics beyond revenue, profit, and assets.

## Contributing

We welcome contributions! Please fork the repository and submit a pull request with your changes. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License.

## Acknowledgements

- Thanks to the developers of `transformers`, `spaCy`, and other libraries used in this project for their amazing work.
- Special thanks to the open-source community for their continuous support and contributions.

---

Feel free to reach out if you have any questions or need further assistance. Happy analyzing!

