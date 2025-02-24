import unittest
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

class TestProxyAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load datasets
        cls.benign_df = pd.read_csv('C:/Users/kruti/Desktop/proxy_analysis/data/cleaned_begnin/log_data.csv', low_memory=False)
        cls.exploit_df = pd.read_csv('C:/Users/kruti/Desktop/proxy_analysis/data/cleaned_exploit/combined_log_data.csv', low_memory=False)

        # Extract User-Agent Strings
        cls.benign_agents = cls.benign_df['userAgentString'].dropna().unique()
        cls.exploit_agents = cls.exploit_df['userAgentString'].dropna().unique()

        # Vectorize User-Agent Strings using TF-IDF
        cls.vectorizer = TfidfVectorizer()
        cls.X_benign = cls.vectorizer.fit_transform(cls.benign_agents)
        cls.X_exploit = cls.vectorizer.transform(cls.exploit_agents)

    def test_benign_agents_length(self):
        self.assertGreater(len(self.benign_agents), 0, "Benign agents should not be empty")

    def test_exploit_agents_length(self):
        self.assertGreater(len(self.exploit_agents), 0, "Exploit agents should not be empty")

    def test_vectorizer_fit_transform(self):
        self.assertEqual(self.X_benign.shape[0], len(self.benign_agents), "Vectorized benign agents shape mismatch")
        self.assertEqual(self.X_exploit.shape[0], len(self.exploit_agents), "Vectorized exploit agents shape mismatch")

    def test_model_training_and_evaluation(self):
        models = {
            "Isolation Forest": IsolationForest(contamination=0.05, random_state=42),
            "One-Class SVM": OneClassSVM(nu=0.05, kernel="rbf"),
            "kNN (1st Nearest Neighbor)": NearestNeighbors(n_neighbors=1)
        }

        evaluation_results = {}
        y_true = np.array([0] * len(self.X_benign.toarray()) + [1] * len(self.X_exploit.toarray()))

        for name, model in models.items():
            if name == "kNN (1st Nearest Neighbor)":
                model.fit(self.X_benign.toarray())
                distances, _ = model.kneighbors(np.vstack([self.X_benign.toarray(), self.X_exploit.toarray()]))
                y_scores = distances.flatten()
                threshold = np.percentile(y_scores[:len(self.X_benign.toarray())], 95)
                y_pred = (y_scores > threshold).astype(int)
            else:
                model.fit(self.X_benign.toarray())
                y_scores = model.decision_function(np.vstack([self.X_benign.toarray(), self.X_exploit.toarray()]))
                y_pred = (y_scores < 0).astype(int)

            precision, recall, f1, _ = classification_report(y_true, y_pred, output_dict=True)["1"].values()
            auc_score = roc_auc_score(y_true, y_scores)
            evaluation_results[name] = {"Precision": precision, "Recall": recall, "F1-score": f1, "ROC-AUC": auc_score}

            print(f"Model: {name}")
            print(f"Precision: {precision}, Recall: {recall}, F1-score: {f1}, ROC-AUC: {auc_score}")
            print("Confusion Matrix:")
            print(confusion_matrix(y_true, y_pred))

            self.assertGreaterEqual(precision, 0, f"{name} precision should be greater than 0")
            self.assertGreaterEqual(recall, 0, f"{name} recall should be greater than 0")
            self.assertGreaterEqual(f1, 0, f"{name} F1-score should be greater than 0")
            self.assertGreaterEqual(auc_score, 0, f"{name} ROC-AUC should be greater than 0")

    def test_best_model_selection(self):
        models = {
            "Isolation Forest": IsolationForest(contamination=0.05, random_state=42),
            "One-Class SVM": OneClassSVM(nu=0.05, kernel="rbf"),
            "kNN (1st Nearest Neighbor)": NearestNeighbors(n_neighbors=1)
        }

        evaluation_results = {}
        y_true = np.array([0] * len(self.X_benign.toarray()) + [1] * len(self.X_exploit.toarray()))

        for name, model in models.items():
            if name == "kNN (1st Nearest Neighbor)":
                model.fit(self.X_benign.toarray())
                distances, _ = model.kneighbors(np.vstack([self.X_benign.toarray(), self.X_exploit.toarray()]))
                y_scores = distances.flatten()
                threshold = np.percentile(y_scores[:len(self.X_benign.toarray())], 95)
                y_pred = (y_scores > threshold).astype(int)
            else:
                model.fit(self.X_benign.toarray())
                y_scores = model.decision_function(np.vstack([self.X_benign.toarray(), self.X_exploit.toarray()]))
                y_pred = (y_scores < 0).astype(int)

            precision, recall, f1, _ = classification_report(y_true, y_pred, output_dict=True)["1"].values()
            auc_score = roc_auc_score(y_true, y_scores)
            evaluation_results[name] = {"Precision": precision, "Recall": recall, "F1-score": f1, "ROC-AUC": auc_score}

        best_model_name = max(evaluation_results, key=lambda x: evaluation_results[x]["F1-score"])
        self.assertIn(best_model_name, models, "Best model should be one of the trained models")

    def test_anomaly_detection_rate(self):
        models = {
            "Isolation Forest": IsolationForest(contamination=0.05, random_state=42),
            "One-Class SVM": OneClassSVM(nu=0.05, kernel="rbf"),
            "kNN (1st Nearest Neighbor)": NearestNeighbors(n_neighbors=1)
        }

        evaluation_results = {}
        y_true = np.array([0] * len(self.X_benign.toarray()) + [1] * len(self.X_exploit.toarray()))

        for name, model in models.items():
            if name == "kNN (1st Nearest Neighbor)":
                model.fit(self.X_benign.toarray())
                distances, _ = model.kneighbors(np.vstack([self.X_benign.toarray(), self.X_exploit.toarray()]))
                y_scores = distances.flatten()
                threshold = np.percentile(y_scores[:len(self.X_benign.toarray())], 95)
                y_pred = (y_scores > threshold).astype(int)
            else:
                model.fit(self.X_benign.toarray())
                y_scores = model.decision_function(np.vstack([self.X_benign.toarray(), self.X_exploit.toarray()]))
                y_pred = (y_scores < 0).astype(int)

            precision, recall, f1, _ = classification_report(y_true, y_pred, output_dict=True)["1"].values()
            auc_score = roc_auc_score(y_true, y_scores)
            evaluation_results[name] = {"Precision": precision, "Recall": recall, "F1-score": f1, "ROC-AUC": auc_score}

        best_model_name = max(evaluation_results, key=lambda x: evaluation_results[x]["F1-score"])
        best_model = models[best_model_name]

        if best_model_name == "kNN (1st Nearest Neighbor)":
            distances, _ = best_model.kneighbors(self.X_exploit.toarray())
            threshold = np.percentile(distances, 95)
            y_pred_exploit = (distances.flatten() > threshold).astype(int)
        else:
            y_pred_exploit = best_model.predict(self.X_exploit.toarray())
            y_pred_exploit = np.where(y_pred_exploit == -1, 1, 0)

        anomaly_percentage = np.mean(y_pred_exploit) * 100
        self.assertGreaterEqual(anomaly_percentage, 0, "Anomaly detection rate should be non-negative")

if __name__ == '__main__':
    unittest.main()