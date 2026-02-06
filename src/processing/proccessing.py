import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


class DataNormalizer:
    def __init__(self):
        self.df = None
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self, filepath):
        self.df = pd.read_csv(filepath)
        print(f"Loaded {len(self.df)} records")
        return self.df

    def drop_columns(self, columns):
        existing_cols = [col for col in columns if col in self.df.columns]
        self.df = self.df.drop(columns=existing_cols)
        return self.df

    def encode_categorical(self):
        if "Age" in self.df.columns:
            self.df["Age_bucket"] = pd.cut(
                self.df["Age"],
                bins=[0, 18, 30, 45, 60, 200],
                labels=["under_18", "18_29", "30_44", "45_59", "60_plus"]
            )
        cat_cols = [
            c for c in ["gender", "attend_group_lesson", "Age_bucket", "drink_abo", "personal_training", "uses_sauna"]
            if c in self.df.columns
        ]

        if cat_cols:
            self.df = pd.get_dummies(self.df, columns=cat_cols, drop_first=True)

        return self.df

    def encode_target(self, target_column='abonoment_type'):
        self.df[target_column] = self.label_encoder.fit_transform(self.df[target_column])
        print(self.df, 'used')
        return self.df

    def split_features_target(self, target_column="abonoment_type"):
        X = self.df.drop(columns=[target_column])
        y = self.df[target_column]
        self.feature_names = X.columns.tolist()
        return X, y

    def train_test_split_data(self, test_size=0.2, random_state=42, stratify=True):
        X, y = self.split_features_target()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if stratify else None
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    def save_training_data(self, output_dir="data/processed"):
        import os
        os.makedirs(output_dir, exist_ok=True)

        np.save(f"{output_dir}/y_train.npy", self.y_train)
        np.save(f"{output_dir}/y_test.npy", self.y_test)


        with open(f"{output_dir}/label_encoder.pkl", "wb") as f:
            pickle.dump(self.label_encoder, f)

        with open(f"{output_dir}/feature_names.pkl", "wb") as f:
            pickle.dump(self.feature_names, f)

        print(f"\nTraining data saved to {output_dir}/")
        print(f"Feature count: {len(self.feature_names)}")

    def load_training_data(self, input_dir="data/processed"):
        self.y_train = np.load(f"{input_dir}/y_train.npy")
        self.y_test = np.load(f"{input_dir}/y_test.npy")

        with open(f"{input_dir}/label_encoder.pkl", "rb") as f:
            self.label_encoder = pickle.load(f)

        with open(f"{input_dir}/feature_names.pkl", "rb") as f:
            self.feature_names = pickle.load(f)

        print(f"Training data loaded from {input_dir}/")
        return self.X_train, self.X_test, self.y_train, self.y_test

    def full_pipeline(self, filepath, target_column="abonoment_type", columns_to_drop=['s', 's'], save_data=False, output_dir="data/processed"):
        self.load_data(filepath)
        print(self.df["abonoment_type"].value_counts())
        print(self.df.groupby("abonoment_type")[[
            "visit_per_week",
            "avg_time_in_gym",
            "personal_training",
            "attend_group_lesson",
            "drink_abo",
            "uses_sauna",

        ]].mean())

        self.drop_columns(columns_to_drop)
        self.df = self.df.fillna(0)

        self.encode_categorical()
        self.encode_target(target_column)

        self.train_test_split_data()

        assert len(self.feature_names) < 30, "Feature explosion detected"
        if save_data:
            self.save_training_data(output_dir)

        return self.X_train, self.X_test, self.y_train, self.y_test, self.feature_names
