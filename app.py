"""
app.py - Fake Job Posting Detection API
Flask backend that loads the trained model and exposes:
  POST /predict  → returns prediction + confidence + suspicious keywords
  GET  /health   → health check for the API
"""

import os
import re
import pickle
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ─────────────────────────────────────────────
#  APP SETUP
# ─────────────────────────────────────────────
app = Flask(__name__, static_folder=".")
CORS(app)  # allow all origins (fine for a college project; restrict in production)

# ─────────────────────────────────────────────
#  LOAD MODEL AT STARTUP
# ─────────────────────────────────────────────
MODEL_PATH = "model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file '{MODEL_PATH}' not found. "
        "Please run: python model.py"
    )

with open(MODEL_PATH, "rb") as f:
    model_data = pickle.load(f)

pipeline           = model_data["pipeline"]
SUSPICIOUS_KEYWORDS = model_data["suspicious_keywords"]

print(f"[INFO] Model loaded from {MODEL_PATH}")


# ─────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    """Mirror the same preprocessing used during training."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def find_suspicious_keywords(raw_text: str) -> list[str]:
    """
    Scan the raw job description for known suspicious phrases.
    Returns a deduplicated list of matched keywords found in the text.
    """
    text_lower = raw_text.lower()
    found = []
    for kw in SUSPICIOUS_KEYWORDS:
        if kw in text_lower and kw not in found:
            found.append(kw)
    return found


def get_risk_level(confidence: float, prediction: int) -> str:
    """Map prediction + confidence to a human-readable risk level."""
    if prediction == 0:
        return "Low"
    if confidence >= 0.85:
        return "High"
    elif confidence >= 0.65:
        return "Medium"
    return "Low"


# ─────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def serve_index():
    """Serve the frontend HTML file directly from Flask."""
    return send_from_directory(".", "index.html")


@app.route("/health", methods=["GET"])
def health():
    """Simple health check — useful for deployment monitoring."""
    return jsonify({"status": "ok", "model_loaded": True}), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Main prediction endpoint.

    Request body (JSON):
        { "text": "<job description>" }

    Response (JSON):
        {
            "prediction":          0 or 1,
            "label":               "Real" or "Fake",
            "confidence":          0.0 – 1.0,
            "risk_level":          "Low" | "Medium" | "High",
            "suspicious_keywords": [ ... ],
            "explanation":         "..."
        }
    """
    # ── Validate request ──────────────────────
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    body = request.get_json()
    raw_text = body.get("text", "").strip()

    if not raw_text:
        return jsonify({"error": "Field 'text' is required and cannot be empty"}), 400

    if len(raw_text) < 20:
        return jsonify({"error": "Job description is too short to analyze (minimum 20 characters)"}), 400

    # ── Preprocess & Predict ──────────────────
    clean = preprocess_text(raw_text)
    proba = pipeline.predict_proba([clean])[0]  # shape: [P(real), P(fake)]
    prediction = int(pipeline.predict([clean])[0])
    confidence = float(proba[prediction])       # confidence for the predicted class

    # ── Explainability ────────────────────────
    suspicious = find_suspicious_keywords(raw_text)
    risk_level = get_risk_level(confidence, prediction)

    if prediction == 1:  # Fake
        if suspicious:
            explanation = (
                f"This posting shows {len(suspicious)} suspicious pattern(s): "
                f"{', '.join(suspicious[:5])}. "
                "These phrases are commonly associated with fraudulent listings."
            )
        else:
            explanation = (
                "The model detected statistical patterns typical of fake postings "
                "based on writing style and word choice, even without obvious red-flag phrases."
            )
    else:  # Real
        if suspicious:
            explanation = (
                f"Mostly legitimate, but contains {len(suspicious)} phrase(s) that appear in some fake postings: "
                f"{', '.join(suspicious[:3])}. Verify independently before applying."
            )
        else:
            explanation = (
                "This posting appears legitimate. The language and structure align "
                "with genuine job advertisements."
            )

    return jsonify({
        "prediction":          prediction,
        "label":               "Fake" if prediction == 1 else "Real",
        "confidence":          round(confidence * 100, 2),   # as percentage
        "risk_level":          risk_level,
        "suspicious_keywords": suspicious,
        "explanation":         explanation,
    }), 200


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("[INFO] Starting Fake Job Detection API on http://127.0.0.1:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
