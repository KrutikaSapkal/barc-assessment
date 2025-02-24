import numpy as np
from fastapi import FastAPI
import uvicorn
import joblib
from sklearn.neighbors import NearestNeighbors
from pydantic import BaseModel

from feature_extraction import vectorize_user_agents
from moder_saver import load_model  # Fixed import name

app = FastAPI()


# Define input model for request body
class UserAgentRequest(BaseModel):
    user_agent: str


@app.post("/predict/")
def predict(request: UserAgentRequest):
    model, vectorizer = load_model()
    X_test = vectorizer.transform([request.user_agent]).toarray()

    if isinstance(model, NearestNeighbors):
        distances, _ = model.kneighbors(X_test)
        threshold = np.percentile(distances, 95)
        prediction = int(distances.flatten() > threshold)
    else:
        prediction = model.predict(X_test)
        prediction = int(prediction == -1)

    return {"user_agent": request.user_agent, "is_anomalous": bool(prediction)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)  # Changed port to 8001
