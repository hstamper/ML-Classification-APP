# 🎫 TicketFlow AI

**Intelligent Customer Support Ticket Classification**

A production-grade deep learning application that automatically classifies customer support tickets into the appropriate department using a hybrid CNN-LSTM neural network. Built with TensorFlow/Keras and served through a Streamlit interface.

---

## Overview

TicketFlow AI routes incoming support tickets to one of four departments in real time:

| Department | Description |
|---|---|
| 🔧 **Technical Support** | Hardware, software, and infrastructure issues |
| 💰 **Sales** | Pricing, plans, and purchase inquiries |
| 📄 **Billing** | Payments, invoices, and account charges |
| 💬 **Customer Service** | General inquiries and feedback |

The system combines a trained CNN-LSTM model with keyword-boosted scoring to deliver accurate classifications with confidence metrics.

---

## Architecture

The classification pipeline operates in three stages:

**1. Text Preprocessing** — Raw ticket text is cleaned (lowercased, URLs/HTML/emails removed, special characters stripped), tokenized using NLTK, stopwords removed (with domain-important words preserved), and lemmatized.

**2. Model Inference** — The preprocessed text is tokenized into sequences, padded to a fixed length of 150 tokens, and fed through a hybrid CNN-LSTM neural network:
- Embedding layer (128 dimensions, 5000-word vocabulary)
- Conv1D layer (64 filters, kernel size 5) for local feature extraction
- MaxPooling1D for dimensionality reduction
- Bidirectional LSTM (64 units) for sequential dependencies
- GlobalMaxPooling1D, Dense layers (128 → 64 → 4), and Softmax output

**3. Keyword Boosting** — Model probabilities are blended with keyword-based signals (60% model / 40% keywords) to improve accuracy on tickets with strong departmental indicators. The keyword dictionary covers domain-specific terms for each department.

---

## Project Structure

```
Ticket_Classification_Project/
├── .streamlit/
│   └── config.toml              # Streamlit theme configuration
├── config/
│   ├── __init__.py
│   └── settings.py              # Paths, department config, app settings
├── data/                        # Data directory
├── models/
│   ├── best_ticket_model.keras  # Best model checkpoint
│   ├── ticket_classifier_model.keras  # Trained CNN-LSTM model
│   ├── ticket_label_encoder.pkl       # Label encoder (4 departments)
│   ├── ticket_preprocessing_params.pkl # Preprocessing parameters
│   └── ticket_tokenizer.pkl           # Fitted tokenizer (5000 words)
├── utils/
│   ├── __init__.py
│   ├── model.py                 # TicketClassifier with keyword boosting
│   └── preprocessing.py         # Text cleaning and lemmatization pipeline
├── assets/                      # Static assets
├── app.py                       # Main Streamlit application
├── requirements.txt             # Python dependencies
└── README.md
```

---

## Setup & Installation

### Prerequisites

- Python 3.10
- macOS (Apple Silicon) or Linux
- Conda (recommended) or pip

### Installation with Conda (Recommended)

```bash
# Create and activate environment
conda create -n ticket_env python=3.10 -y
conda activate ticket_env

# Install dependencies
conda install scikit-learn tensorflow numpy pandas nltk -c conda-forge -y
conda install pyarrow -y
pip install streamlit blinker
```

### Installation with pip

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Running the Application

```bash
cd Ticket_Classification_Project
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

---

## Features

### ⚡ Single Ticket Classification

Paste or type a support ticket to get instant department routing with confidence scores and probability distributions across all four departments. Includes sample tickets for quick testing.

### 📦 Batch Classification

Upload a CSV file with a `ticket_text` column to classify hundreds of tickets at once. Provides summary statistics, department breakdowns, and a downloadable results CSV with predictions and confidence scores.

### 📋 Classification History

All classifications within a session are logged with timestamps, department assignments, and confidence levels. View a running history of all tickets processed.

### 📊 Analytics Dashboard

Track classification patterns with department distribution charts, confidence trends over time, and aggregate statistics including average, minimum, and maximum confidence scores.

---

## Model Performance

The CNN-LSTM hybrid model was trained on a labeled dataset of customer support tickets with the following characteristics:

- **Training split**: 80/20 train/test with stratified sampling
- **Optimizer**: Adam with ReduceLROnPlateau scheduling
- **Early stopping**: Patience of 5 epochs monitoring validation loss
- **Loss function**: Categorical cross-entropy

The keyword boosting layer further improves real-world accuracy by reinforcing domain-specific vocabulary signals that the model may underweight.

---

## Configuration

### Adjusting Keyword Boost Weights

In `utils/model.py`, modify the model/keyword blend ratio:

```python
# Current: 60% model + 40% keyword boost
combined = (model_score * 0.60) + (keyword_score * 0.40)
```

Increase the keyword weight if classifications are too dependent on the model, or decrease it if keyword signals are overriding correct model predictions.

### Adding Keywords

In `utils/model.py`, update the `DEPARTMENT_KEYWORDS` dictionary. Strong keywords contribute 0.15 per match, moderate keywords contribute 0.05:

```python
DEPARTMENT_KEYWORDS = {
    "Billing": {
        "strong": ["charge", "refund", "invoice", ...],
        "moderate": ["money", "pay", "account", ...],
    },
    ...
}
```

### Theme Customization

Edit `.streamlit/config.toml` to modify the dark theme colors, or update the CSS variables in `app.py` under the `:root` selector.

---

## Technology Stack

- **Deep Learning**: TensorFlow / Keras (CNN-LSTM hybrid architecture)
- **NLP**: NLTK (tokenization, lemmatization, stopword removal)
- **ML Utilities**: scikit-learn (label encoding, metrics)
- **Frontend**: Streamlit with custom dark industrial theme
- **Data Processing**: pandas, NumPy

---

## License

This project is intended for educational and internal use.
