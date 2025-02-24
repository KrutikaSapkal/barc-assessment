import joblib

def save_model(model, vectorizer):
    joblib.dump(model, "best_model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")

def load_model():
    return joblib.load("best_model.pkl"), joblib.load("vectorizer.pkl")
