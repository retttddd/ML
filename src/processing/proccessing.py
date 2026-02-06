import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


class DataNormalizer:
    def __init__(self):
        self.df = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None

    def load_data(self, filepath):
        self.df = pd.read_csv(filepath)
        print(f"Loaded {len(self.df)} records")
        return self.df

    def preview_data(self):
        print("\n=== Dataset Preview ===")
        print(self.df.head())
        print("\n=== Data Types ===")
        print(self.df.dtypes)

    def extract_days_count(self):
        if 'days_per_week' in self.df.columns:
            self.df['num_days_per_week'] = self.df['days_per_week'].str.split(',').str.len()
        return self.df

    def drop_columns(self, columns):
        existing_cols = [col for col in columns if col in self.df.columns]
        self.df = self.df.drop(columns=existing_cols)
        return self.df

    def handle_missing_values(self):
        ## still vibe coded meth
        num_cols = self.df.select_dtypes(include=["int64", "float64"]).columns
        self.df[num_cols] = self.df[num_cols].fillna(self.df[num_cols].median())

        cat_cols = self.df.select_dtypes(include=["object"]).columns
        for col in cat_cols:
            if len(self.df[col].mode()) > 0:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])

        return self.df

    def encode_categorical(self, target_column='abonoment_type'):
        cat_cols = self.df.select_dtypes(include=["object"]).columns.tolist()

        if target_column in cat_cols:
            cat_cols.remove(target_column)

        if cat_cols:
            self.df = pd.get_dummies(self.df, columns=cat_cols, drop_first=True)

        return self.df

    def encode_target(self, target_column='abonoment_type'):
        self.df[target_column] = self.label_encoder.fit_transform(self.df[target_column])
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

    def normalize_data(self):
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        return self.X_train_scaled, self.X_test_scaled

    def save_training_data(self, output_dir="data/processed"):
        import os
        os.makedirs(output_dir, exist_ok=True)

        np.save(f"{output_dir}/X_train_scaled.npy", self.X_train_scaled)
        np.save(f"{output_dir}/X_test_scaled.npy", self.X_test_scaled)
        np.save(f"{output_dir}/y_train.npy", self.y_train)
        np.save(f"{output_dir}/y_test.npy", self.y_test)

        with open(f"{output_dir}/scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)

        with open(f"{output_dir}/label_encoder.pkl", "wb") as f:
            pickle.dump(self.label_encoder, f)

        with open(f"{output_dir}/feature_names.pkl", "wb") as f:
            pickle.dump(self.feature_names, f)

        print(f"\nTraining data saved to {output_dir}/")
        print(f"Feature count: {len(self.feature_names)}")

    def load_training_data(self, input_dir="data/processed"):
        self.X_train_scaled = np.load(f"{input_dir}/X_train_scaled.npy")
        self.X_test_scaled = np.load(f"{input_dir}/X_test_scaled.npy")
        self.y_train = np.load(f"{input_dir}/y_train.npy")
        self.y_test = np.load(f"{input_dir}/y_test.npy")

        with open(f"{input_dir}/scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)

        with open(f"{input_dir}/label_encoder.pkl", "rb") as f:
            self.label_encoder = pickle.load(f)

        with open(f"{input_dir}/feature_names.pkl", "rb") as f:
            self.feature_names = pickle.load(f)

        print(f"Training data loaded from {input_dir}/")
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test

    def full_pipeline(self, filepath, target_column="abonoment_type",
                     columns_to_drop=["id", "name_personal_trainer", "birthday", "days_per_week"],
                     save_data=False, output_dir="data/processed"):
        print("=== Starting Data Processing Pipeline ===\n")

        self.load_data(filepath)
        self.extract_days_count()
        self.preview_data()
        self.drop_columns(columns_to_drop)
        self.handle_missing_values()
        self.encode_categorical(target_column)
        self.encode_target(target_column)
        self.train_test_split_data()
        self.normalize_data()

        print("\n=== Pipeline Complete ===")
        print(f"X_train shape: {self.X_train_scaled.shape}")
        print(f"X_test shape: {self.X_test_scaled.shape}")
        print(f"Target classes: {self.label_encoder.classes_}")

        if save_data:
            self.save_training_data(output_dir)

        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test, self.feature_names
