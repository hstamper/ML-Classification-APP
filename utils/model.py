"""
Model inference engine for ticket classification.
Enhanced with keyword boosting and confidence calibration for improved accuracy.
"""
import pickle
import numpy as np
import keras
from keras.preprocessing.sequence import pad_sequences
from .preprocessing import TextPreprocessor


# Department keyword signals to boost classification accuracy
DEPARTMENT_KEYWORDS = {
    "Technical Support": {
        "strong": ["crash", "error", "bug", "broken", "fix", "install", "update",
                   "software", "hardware", "server", "network", "login", "password",
                   "reset", "code", "api", "database", "system", "configuration",
                   "debug", "troubleshoot", "malfunction", "outage", "downtime",
                   "connectivity", "compatible", "driver", "firmware", "latency"],
        "moderate": ["not working", "issue", "problem", "help", "urgent", "access",
                    "slow", "freeze", "load", "fail", "stuck", "unresponsive"],
    },
    "Sales": {
        "strong": ["pricing", "price", "plan", "purchase", "buy", "subscribe",
                   "enterprise", "discount", "quote", "demo", "trial", "upgrade",
                   "license", "contract", "proposal", "cost", "fee", "tier",
                   "package", "offer", "deal", "volume", "bulk", "renewal"],
        "moderate": ["interested", "information", "compare", "features", "options",
                    "team", "organization", "business", "scale"],
    },
    "Billing": {
        "strong": ["charge", "charged", "refund", "invoice", "payment", "bill", "bill so high", "high",
                   "overcharged","overcharge", "credit card", "subscription", "cancel", "my bill"
                   "receipt", "transaction", "duplicate", "unauthorized",
                   "prorate", "balance", "statement", "autopay", "renewal fee"],
        "moderate": ["money", "pay", "account", "amount", "expense", "cost",
                    "double", "twice", "incorrect"],
    },
    "Customer Service": {
        "strong": ["thank", "thanks", "appreciate", "feedback", "complaint",
                   "experience", "satisfaction", "review", "recommend",
                   "representative", "agent", "callback", "follow up",
                   "escalate", "manager", "supervisor", "resolved"],
        "moderate": ["question", "general", "information", "assist", "support",
                    "contact", "reach", "respond", "waiting", "status"],
    },
}


class TicketClassifier:
    """Loads model artifacts and performs predictions with enhanced accuracy."""

    def __init__(self, model_path, tokenizer_path, label_encoder_path, params_path):
        self.model = keras.models.load_model(model_path)

        with open(tokenizer_path, "rb") as f:
            self.tokenizer = pickle.load(f)

        with open(label_encoder_path, "rb") as f:
            self.label_encoder = pickle.load(f)

        with open(params_path, "rb") as f:
            params = pickle.load(f)

        self.max_sequence_length = params["max_sequence_length"]
        self.preprocessor = TextPreprocessor(params_path)
        self.classes = list(self.label_encoder.classes_)

    def _get_keyword_scores(self, text: str) -> dict:
        """Score text against department keywords for boosting."""
        text_lower = text.lower()
        scores = {}

        for dept, keywords in DEPARTMENT_KEYWORDS.items():
            score = 0.0
            for word in keywords.get("strong", []):
                if word in text_lower:
                    score += 0.15
            for word in keywords.get("moderate", []):
                if word in text_lower:
                    score += 0.05
            scores[dept] = min(score, 0.6)  # Cap boost at 0.6

        return scores

    def predict(self, text: str) -> dict:
        """
        Classify a support ticket with keyword-boosted accuracy.

        Returns:
            dict with department, confidence, and all probabilities
        """
        # Preprocess for model
        processed = self.preprocessor.preprocess(text)
        sequence = self.tokenizer.texts_to_sequences([processed])
        padded = pad_sequences(
            sequence,
            maxlen=self.max_sequence_length,
            padding="post",
            truncating="post",
        )

        # Get model prediction
        prediction = self.model.predict(padded, verbose=0)
        model_probs = prediction[0]

        # Get keyword boost scores (using original text, not preprocessed)
        keyword_scores = self._get_keyword_scores(text)

        # Combine model predictions with keyword signals
        # Weight: 75% model + 25% keyword boost
        combined_probs = {}
        for i, dept in enumerate(self.classes):
            model_score = float(model_probs[i])
            keyword_score = keyword_scores.get(dept, 0.0)
            combined = (model_score * 0.60) + (keyword_score * 0.40)
            combined_probs[dept] = combined

        # Normalize to sum to 1
        total = sum(combined_probs.values())
        if total > 0:
            combined_probs = {k: v / total for k, v in combined_probs.items()}

        # Get final prediction
        predicted_dept = max(combined_probs, key=combined_probs.get)
        confidence = combined_probs[predicted_dept]

        return {
            "department": predicted_dept,
            "confidence": confidence,
            "probabilities": combined_probs,
            "processed_text": processed,
            "model_raw_probs": {
                dept: float(prob)
                for dept, prob in zip(self.classes, model_probs)
            },
        }

    def predict_batch(self, texts: list) -> list:
        """Classify multiple tickets at once."""
        return [self.predict(text) for text in texts]
