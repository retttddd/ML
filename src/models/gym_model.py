import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class AbonementPredictor:
    def __init__(self, model_path="models/gym_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.encoder = None
        self.feature_names = None

    # ---------- Training ----------
    def train(self, X_train, y_train):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        self.model = RandomForestClassifier(
            random_state=42,
            max_depth=15,
            min_samples_split=5
        )

        self.model.fit(X_train, y_train)
        return self.model

    # ---------- Persistence ----------
    def save(self):
        if not self.model:
            raise ValueError("Train model before saving")

        with open(self.model_path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self):
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)

        return self.model

    def load_preprocessing(self, path="data/processed"):

        with open(f"{path}/label_encoder.pkl", "rb") as f:
            self.encoder = pickle.load(f)

        with open(f"{path}/feature_names.pkl", "rb") as f:
            self.feature_names = pickle.load(f)

    # ---------- Prediction ----------
    def _prepare_input(self, data: dict):
        df = pd.DataFrame([data])
        df = pd.get_dummies(df)

        for col in self.feature_names:
            if col not in df:
                df[col] = 0

        df = df[self.feature_names]
        return df

    def predict(self, data: dict):
        if not all([self.model, self.encoder, self.feature_names]):
            raise ValueError("Load model and preprocessing first")

        X = self._prepare_input(data)

        pred = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]

        return {
            "abonement_type": self.encoder.inverse_transform([pred])[0],
            "confidence": float(max(proba)),
        }

    # ---------- Evaluation ----------
    def evaluate(self, X_test, y_test):
        preds = self.model.predict(X_test)

        print("Accuracy:", accuracy_score(y_test, preds))
        print(classification_report(y_test, preds, target_names=self.encoder.classes_))
        print("Confusion matrix:\n", confusion_matrix(y_test, preds))
