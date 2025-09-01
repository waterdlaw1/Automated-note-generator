📝 AI Note Generator

Transform documents into comprehensive notes with AI-powered summarization. This Streamlit application processes PDF, DOCX, and text files to generate executive summaries, key points, and action items.

[Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
[Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
[Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)

✨ Features

- Multiple File Support: Process PDF, DOCX, and TXT files
- AI-Powered Summarization: Utilizes state-of-the-art transformer models
- Text Analysis: Get detailed statistics about your documents
- Key Phrase Extraction: Automatically identify important concepts
- Smart Chunking: Handles large documents efficiently
- Fallback Mechanism: Works even when API is unavailable
- Export Functionality: Download notes as text files
- Responsive UI: Beautiful interface with custom styling

🚀 Quick Start

Prerequisites

- Python 3.7+
- Hugging Face API key ([Get one here](https://huggingface.co/settings/tokens))

Installation

1. Clone the repository:
bash
git clone https://github.com/yourusername/ai-note-generator.git
cd ai-note-generator
Install dependencies:

bash
pip install -r requirements.txt
Run the application:

bash
streamlit run app.py
Open your browser and navigate to the local URL shown in the terminal

🔧 Configuration
On first run, the app will prompt you to enter your Hugging Face API key. This will be saved in a .env file for future use.

Available Models
BART Large CNN (default)

DistilBART CNN

Pegasus XSum

T5 Small

Advanced Settings
Summary Length: Control the length of generated summaries (50-500 words)

Chunk Size: Adjust how large documents are split for processing

API Timeout: Set timeout duration for API requests

Max Retries: Configure retry attempts for failed requests

📖 Usage
Upload a document (PDF, DOCX, or TXT) or paste text directly

Adjust settings in the sidebar if needed

Click "Generate Comprehensive Notes"

View results in the organized tabs:

Executive Summary

Key Points

Action Items

Download your notes as a text file

🛠️ Technical Details
How It Works
Text Extraction: The app uses pdfplumber for PDFs and python-docx for Word documents

Text Cleaning: Removes excessive whitespace and normalizes text

Complexity Analysis: Calculates word count, reading time, and other metrics

Summarization: Sends text to Hugging Face's inference API

Result Processing: Formats the output and extracts key information

Error Handling
The application includes comprehensive error handling with:

API timeout management

Retry logic for failed requests

Fallback to simple summarization when API is unavailable

User-friendly error messages

📊 Sample Output
The app generates comprehensive notes including:

Executive summary

Bulleted key points

Action items

Text statistics (word count, reading time, etc.)

Key phrases extracted from the document

🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
Built with Streamlit

Uses models from Hugging Face

Icons by Twemoji

text

This README provides a comprehensive overview of the application, its features, installation instructions, and usage guidelines. It's formatted for GitHub with badges, clear sections, and code blocks for easy reading.
