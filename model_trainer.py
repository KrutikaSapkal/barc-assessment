import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import classification_report, roc_auc_score


def train_models(X_benign, X_exploit):
    models = {
        "Isolation Forest": IsolationForest(contamination=0.05, random_state=42),
        "One-Class SVM": OneClassSVM(nu=0.05, kernel="rbf"),
        "kNN (1st Nearest Neighbor)": NearestNeighbors(n_neighbors=1)
    }

    evaluation_results = {}
    y_true = np.array([0] * len(X_benign.toarray()) + [1] * len(X_exploit.toarray()))

    for name, model in models.items():
        if name == "kNN (1st Nearest Neighbor)":
            model.fit(X_benign.toarray())
            distances, _ = model.kneighbors(np.vstack([X_benign.toarray(), X_exploit.toarray()]))
            y_scores = distances.flatten()
            threshold = np.percentile(y_scores[:len(X_benign.toarray())], 95)
            y_pred = (y_scores > threshold).astype(int)
        else:
            model.fit(X_benign.toarray())
            y_scores = model.decision_function(np.vstack([X_benign.toarray(), X_exploit.toarray()]))
            y_pred = (y_scores < 0).astype(int)

        precision, recall, f1, _ = classification_report(y_true, y_pred, output_dict=True)["1"].values()
        auc_score = roc_auc_score(y_true, y_scores)
        evaluation_results[name] = {"Precision": precision, "Recall": recall, "F1-score": f1, "ROC-AUC": auc_score}
        print(f" {name} - Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}, AUC: {auc_score:.2f}")

    best_model_name = max(evaluation_results, key=lambda x: evaluation_results[x]["F1-score"])
    best_model = models[best_model_name]
    print(f"🏆 Best Model Selected: {best_model_name}")

    # Compute Anomaly Percentage on Exploit Data
    if best_model_name == "kNN (1st Nearest Neighbor)":
        distances, _ = best_model.kneighbors(X_exploit.toarray())
        threshold = np.percentile(distances, 95)
        y_pred_exploit = (distances.flatten() > threshold).astype(int)
    else:
        y_pred_exploit = best_model.predict(X_exploit.toarray())
        y_pred_exploit = np.where(y_pred_exploit == -1, 1, 0)

    anomaly_percentage = np.mean(y_pred_exploit) * 100
    print(f" Anomaly Detection Rate in Exploit Data: {anomaly_percentage:.2f}%")

    return best_model, best_model_name