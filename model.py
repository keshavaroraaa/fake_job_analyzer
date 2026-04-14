"""
model.py - Fake Job Posting Detection
Trains a TF-IDF + Logistic Regression classifier on the fake job postings dataset.
Saves the trained model and vectorizer for use in the Flask API.

"""

import pandas as pd
import numpy as np
import pickle
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline


# ─────────────────────────────────────────────
#  1. SUSPICIOUS KEYWORDS (used in explainability)
# ─────────────────────────────────────────────
SUSPICIOUS_KEYWORDS = [
    "earn", "no experience", "work from home", "easy money", "guaranteed",
    "unlimited income", "be your own boss", "make money fast", "no degree",
    "huge income", "passive income", "weekly pay", "daily pay", "wire transfer",
    "western union", "upfront fee", "registration fee", "training fee",
    "multi-level", "mlm", "pyramid", "referral bonus", "100% remote",
    "immediate hire", "urgent hiring", "whatsapp", "telegram", "click here",
    "limited time", "apply now", "no interview", "start today", "$$",
    "part time", "flexible hours", "uncapped commission", "unlimited earning",
    "data entry", "stuffing envelopes", "home based", "work at home",
]


# ─────────────────────────────────────────────
#  2. TEXT PREPROCESSING
# ─────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    """
    Clean and normalize raw job posting text.
    Steps: lowercase → remove HTML/URLs/emails → remove punctuation → collapse spaces
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)          # strip HTML tags
    text = re.sub(r"http\S+|www\.\S+", " ", text)  # remove URLs
    text = re.sub(r"\S+@\S+", " ", text)            # remove emails
    text = re.sub(r"[^a-z0-9\s]", " ", text)        # keep alphanumeric only
    text = re.sub(r"\s+", " ", text).strip()         # collapse whitespace
    return text


def combine_fields(row: pd.Series) -> str:
    """
    Combine multiple columns into one rich text blob.
    Using title + description + requirements gives the model more signal.
    """
    parts = [
        str(row.get("title", "")),
        str(row.get("company_profile", "")),
        str(row.get("description", "")),
        str(row.get("requirements", "")),
        str(row.get("benefits", "")),
        str(row.get("employment_type", "")),
        str(row.get("required_experience", "")),
        str(row.get("required_education", "")),
        str(row.get("industry", "")),
        str(row.get("function", "")),
    ]
    return " ".join(parts)


# ─────────────────────────────────────────────
#  3. SYNTHETIC DATA FALLBACK
#     Used when the Kaggle CSV is not available.
#     Provides enough samples to demonstrate the pipeline.
# ─────────────────────────────────────────────
def generate_synthetic_data() -> pd.DataFrame:
    """Return a small but realistic synthetic dataset."""
    real_jobs = [
        {
            "text": "Software Engineer at Google. We are seeking a talented software engineer with 3+ years of Python experience. "
                    "Responsibilities include designing scalable systems, code reviews, and mentoring junior developers. "
                    "Requirements: BS in Computer Science or equivalent. Competitive salary and benefits offered. "
                    "We are an equal opportunity employer committed to diversity.",
            "fraudulent": 0,
        },
        {
            "text": "Data Analyst at Deloitte. Join our analytics team to drive data-driven decision making. "
                    "You will work with SQL, Tableau, and Python to analyze large datasets. "
                    "5+ years experience required. Full health, dental, and 401k package. Apply via our careers portal.",
            "fraudulent": 0,
        },
        {
            "text": "Marketing Manager at Unilever. Lead integrated marketing campaigns across digital and traditional channels. "
                    "MBA preferred with 7 years of FMCG experience. Travel required 20% of time. "
                    "Excellent communication and leadership skills essential.",
            "fraudulent": 0,
        },
        {
            "text": "Registered Nurse, ICU — Boston Medical Center. Full-time night shifts available. "
                    "BSN required, RN license in Massachusetts mandatory. 2+ years ICU experience preferred. "
                    "Comprehensive benefits including tuition reimbursement and malpractice coverage.",
            "fraudulent": 0,
        },
        {
            "text": "Financial Analyst at JPMorgan Chase. Analyze financial statements, build valuation models, "
                    "and support M&A transactions. CFA candidacy preferred. Excel and Bloomberg proficiency required. "
                    "Strong analytical skills and attention to detail.",
            "fraudulent": 0,
        },
        {
            "text": "UX Designer at Airbnb. Design intuitive user experiences for web and mobile platforms. "
                    "Portfolio demonstrating user-centered design process required. Proficiency in Figma and Sketch. "
                    "Collaborate with cross-functional teams. 4+ years experience. Relocation assistance provided.",
            "fraudulent": 0,
        },
        {
            "text": "Product Manager at Amazon. Drive product roadmap and strategy for our AWS services team. "
                    "Technical background required with MBA preferred. Experience with agile methodologies. "
                    "Exceptional communication skills to align stakeholders across global teams.",
            "fraudulent": 0,
        },
        {
            "text": "Civil Engineer at AECOM. Manage infrastructure projects from concept through construction. "
                    "PE license required. Proficiency in AutoCAD and Civil 3D. "
                    "Strong project management skills. Travel to project sites required.",
            "fraudulent": 0,
        },
        {
            "text": "High School Math Teacher — Chicago Public Schools. Teach Algebra and Geometry to grades 9-12. "
                    "Illinois teaching certificate required. Passion for student success essential. "
                    "Pension plan, summers off, and professional development budget included.",
            "fraudulent": 0,
        },
        {
            "text": "Operations Manager at FedEx. Oversee daily warehouse operations including shipping, receiving, and inventory. "
                    "5+ years logistics experience. Six Sigma certification a plus. "
                    "Must be available for early morning shifts. Competitive pay and profit sharing.",
            "fraudulent": 0,
        },
        {
            "text": "HR Business Partner at Microsoft. Partner with business leaders to develop people strategies. "
                    "SHRM certification preferred. 6+ years HR generalist experience. "
                    "Experience with Workday HRIS system. Strong conflict resolution skills.",
            "fraudulent": 0,
        },
        {
            "text": "Cybersecurity Analyst at Lockheed Martin. Monitor and respond to security incidents. "
                    "Secret clearance required. CISSP or CEH certification preferred. "
                    "Experience with SIEM tools and threat intelligence platforms.",
            "fraudulent": 0,
        },
        {
            "text": "Pharmacist — CVS Health. Dispense medications, counsel patients, and manage pharmacy operations. "
                    "PharmD required. Active state pharmacist license mandatory. "
                    "Excellent customer service skills. Rotating weekend shifts required.",
            "fraudulent": 0,
        },
        {
            "text": "Supply Chain Coordinator at Procter and Gamble. Coordinate supplier relationships and inventory levels. "
                    "Bachelor's degree in supply chain or business required. APICS certification a plus. "
                    "Strong Excel and SAP skills. Attention to detail critical.",
            "fraudulent": 0,
        },
        {
            "text": "Graphic Designer at Nike. Create compelling visual content for global marketing campaigns. "
                    "BFA in Graphic Design required. Expert proficiency in Adobe Creative Suite. "
                    "Strong portfolio demonstrating brand consistency. 3+ years agency or in-house experience.",
            "fraudulent": 0,
        },
        # ── FAKE JOBS ──
        {
            "text": "URGENT HIRING! Earn $500-$2000 per week working from home! No experience needed! "
                    "No degree required! Be your own boss with flexible hours! "
                    "Guaranteed daily pay via wire transfer or western union! "
                    "Click here and apply now! Start TODAY! Limited spots available! WhatsApp us immediately!",
            "fraudulent": 1,
        },
        {
            "text": "Make money fast with our amazing data entry jobs! Work at home earn unlimited income! "
                    "No interview needed, immediate hire! Passive income opportunity of a lifetime! "
                    "Pay a small registration fee of $50 to get your starter kit. "
                    "Uncapped commission, weekly pay guaranteed. 100% remote, apply now!",
            "fraudulent": 1,
        },
        {
            "text": "JOIN OUR TEAM and earn big bucks stuffing envelopes from home! "
                    "Easy money, no experience required! Earn $$$$ daily working part time flexible hours! "
                    "Multi-level marketing opportunity with huge referral bonus! "
                    "Send us a message on Telegram. Be your own boss today!",
            "fraudulent": 1,
        },
        {
            "text": "Work from home assembling products! Make up to $1500 a week with no experience! "
                    "Home based job, guaranteed income every week! Upfront fee of $99 required for training materials. "
                    "Unlimited earning potential! No degree no interview! Start immediately, do not miss this!",
            "fraudulent": 1,
        },
        {
            "text": "AMAZING OPPORTUNITY! Be your own boss! Earn passive income with our proven system! "
                    "Data entry from home, no skills needed! $200-$500 per day guaranteed! "
                    "We will pay via western union or paypal! Apply now limited time offer! "
                    "WhatsApp +1234567890 for details!",
            "fraudulent": 1,
        },
        {
            "text": "Urgent: Online survey taker needed! Earn $50 per survey, no experience required! "
                    "Work from home on your own schedule! Easy money daily pay! "
                    "Just pay a $25 activation fee to get started! Unlimited income! Click here to apply!",
            "fraudulent": 1,
        },
        {
            "text": "Financial freedom is here! Join our multi-level marketing team and earn huge income! "
                    "Pyramid structure with amazing referral bonus! No degree needed! "
                    "Be your own boss, flexible hours! Work from home! Make money fast today!",
            "fraudulent": 1,
        },
        {
            "text": "Earn thousands per month doing simple tasks online! No experience no interview! "
                    "We are urgent hiring! Get paid weekly via wire transfer! "
                    "Pay $75 training fee and start earning immediately! 100% remote guaranteed job!",
            "fraudulent": 1,
        },
        {
            "text": "Work at home mom opportunity! Earn easy money sorting emails and doing data entry! "
                    "No experience required, flexible hours, part time work from home! "
                    "Uncapped commission plus weekly pay bonus! Apply now do not miss out! Message on whatsapp!",
            "fraudulent": 1,
        },
        {
            "text": "MAKE BIG MONEY NOW from the comfort of your home! No skills needed, no degree, no experience! "
                    "100% guaranteed income every day! Simply pay a one time fee of $150 and unlock your earning potential! "
                    "Immediate hire, start today! Western union payment available! APPLY IMMEDIATELY!",
            "fraudulent": 1,
        },
        {
            "text": "Home based business opportunity! Resell our products and earn 500% profit! "
                    "No experience needed, be your own boss! Easy money, daily pay via paypal! "
                    "Referral bonus for every person you bring! Limited time offer, click here now!",
            "fraudulent": 1,
        },
        {
            "text": "Part time work from home! Earn $80 per hour testing websites! No experience required! "
                    "Flexible hours, immediate start, no interview! Pay $30 registration fee to access job portal! "
                    "Guaranteed weekly payment! Message us on Telegram today!",
            "fraudulent": 1,
        },
    ]

    df = pd.DataFrame(real_jobs)
    # Augment by duplicating with slight variations for more training signal
    augmented = []
    for _, row in df.iterrows():
        augmented.append(row)
        # simple augment: shuffle words slightly by repeating text
        new_row = row.copy()
        new_row["text"] = row["text"] + " " + row["text"][:len(row["text"]) // 3]
        augmented.append(new_row)

    df_aug = pd.DataFrame(augmented).reset_index(drop=True)
    return df_aug


# ─────────────────────────────────────────────
#  4. LOAD DATASET
# ─────────────────────────────────────────────
def load_dataset(csv_path: str = "fake_job_postings.csv") -> pd.DataFrame:
    """
    Try to load the Kaggle dataset. Fall back to synthetic data if not found.
    Kaggle dataset: https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction
    """
    try:
        df = pd.read_csv(csv_path)
        print(f"[INFO] Loaded Kaggle dataset: {len(df)} rows")

        # Combine all text fields into one column
        df["text"] = df.apply(combine_fields, axis=1)
        df["text"] = df["text"].apply(preprocess_text)

        # Drop rows with empty text or missing label
        df = df[["text", "fraudulent"]].dropna()
        df = df[df["text"].str.strip() != ""]
        print(f"[INFO] After cleaning: {len(df)} rows | Fake: {df['fraudulent'].sum()} | Real: {(df['fraudulent']==0).sum()}")
        return df

    except FileNotFoundError:
        print("[WARN] Kaggle CSV not found. Using synthetic dataset for demonstration.")
        print("[INFO] Download dataset from: https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction")
        df = generate_synthetic_data()
        df["text"] = df["text"].apply(preprocess_text)
        print(f"[INFO] Synthetic dataset: {len(df)} rows | Fake: {df['fraudulent'].sum()} | Real: {(df['fraudulent']==0).sum()}")
        return df


# ─────────────────────────────────────────────
#  5. BUILD + TRAIN PIPELINE
# ─────────────────────────────────────────────
def build_pipeline() -> Pipeline:
    """
    Sklearn Pipeline:
      TF-IDF → captures word importance across documents
      LogisticRegression → fast, interpretable, works well with sparse text features
      class_weight='balanced' → handles the class imbalance (fewer fake than real jobs)
    """
    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),   # unigrams + bigrams catch "no experience", "work from home"
                max_features=20000,   # vocabulary cap to prevent overfitting
                sublinear_tf=True,    # apply log normalization to TF
                min_df=1,             # keep terms that appear at least once
                strip_accents="unicode",
                analyzer="word",
            ),
        ),
        (
            "clf",
            LogisticRegression(
                class_weight="balanced",  # compensate for imbalanced dataset
                max_iter=1000,
                C=1.0,                    # regularization strength
                solver="lbfgs",
                random_state=42,
            ),
        ),
    ])
    return pipeline


# ─────────────────────────────────────────────
#  6. EVALUATE MODEL
# ─────────────────────────────────────────────
def evaluate(pipeline: Pipeline, X_test: pd.Series, y_test: pd.Series) -> None:
    """Print accuracy, classification report, and confusion matrix."""
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 55)
    print(f"  MODEL EVALUATION")
    print("=" * 55)
    print(f"  Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))

    cm = confusion_matrix(y_test, y_pred)
    print("  Confusion Matrix:")
    print(f"              Predicted Real  Predicted Fake")
    print(f"  Actual Real     {cm[0][0]:>6}          {cm[0][1]:>6}")
    print(f"  Actual Fake     {cm[1][0]:>6}          {cm[1][1]:>6}")
    print("=" * 55 + "\n")


# ─────────────────────────────────────────────
#  7. MAIN — TRAIN & SAVE
# ─────────────────────────────────────────────
def main():
    # Load data
    df = load_dataset("fake_job_postings.csv")

    X = df["text"]
    y = df["fraudulent"]

    # Split — stratified to preserve class ratio in both splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[INFO] Train: {len(X_train)} | Test: {len(X_test)}")

    # Build and train
    print("[INFO] Training model...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    print("[INFO] Training complete.")

    # Evaluate
    evaluate(pipeline, X_test, y_test)

    # Save model to disk
    model_data = {
        "pipeline": pipeline,
        "suspicious_keywords": SUSPICIOUS_KEYWORDS,
    }
    with open("model.pkl", "wb") as f:
        pickle.dump(model_data, f)

    print("[INFO] Model saved to model.pkl")
    print("[INFO] Run 'python app.py' to start the Flask API.")


if __name__ == "__main__":
    main()
