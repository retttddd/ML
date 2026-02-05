import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataNormalizer:
    def __init__(self):
        self.df = None
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None

    def load_data(self, filepath):
        """Load data from CSV file"""
        self.df = pd.read_csv(filepath)
        return self.df

    def preview_data(self):
        """Display basic information about the dataset"""
        print(self.df.head())
        print(self.df.dtypes)

    def create_target(self, column, bins, labels):
        """Create target variable using binning"""
        self.df["visitor_level"] = pd.cut(
            self.df[column],
            bins=bins,
            labels=labels
        )
        return self.df

    def drop_columns(self, columns):
        self.df = self.df.drop(columns=columns)
        return self.df

    def handle_missing_values(self):
        ## vibe coded shit. TODO: Rewrite, not sure about this method
        num_cols = self.df.select_dtypes(include=["int64", "float64"]).columns
        self.df[num_cols] = self.df[num_cols].fillna(self.df[num_cols].median())

        cat_cols = self.df.select_dtypes(include=["object"]).columns
        for col in cat_cols:
            self.df[col] = self.df[col].fillna(self.df[col].mode()[0])

        return self.df

    def encode_categorical(self):
        """Perform one-hot encoding on categorical columns"""
        cat_cols = self.df.select_dtypes(include=["object"]).columns
        self.df = pd.get_dummies(self.df, columns=cat_cols, drop_first=True)
        return self.df

    def split_features_target(self, target_column="visitor_level"):
        """Split data into features (X) and target (y)"""
        X = self.df.drop(columns=[target_column])
        y = self.df[target_column]
        return X, y

    def train_test_split_data(self, test_size=0.2, random_state=42, stratify=True):
        """Split data into training and testing sets"""
        X, y = self.split_features_target()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if stratify else None
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    def normalize_data(self):
        """Normalize training and testing data using StandardScaler"""
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        return self.X_train_scaled, self.X_test_scaled

    def save_training_data(self, output_dir="data/processed"):
        """Save processed training data and scaler"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Save as numpy arrays
        np.save(f"{output_dir}/X_train_scaled.npy", self.X_train_scaled)
        np.save(f"{output_dir}/X_test_scaled.npy", self.X_test_scaled)
        np.save(f"{output_dir}/y_train.npy", self.y_train)
        np.save(f"{output_dir}/y_test.npy", self.y_test)

        # Save the scaler
        with open(f"{output_dir}/scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)

        print(f"Training data saved to {output_dir}/")

    def load_training_data(self, input_dir="data/processed"):
        """Load previously saved training data and scaler"""
        self.X_train_scaled = np.load(f"{input_dir}/X_train_scaled.npy")
        self.X_test_scaled = np.load(f"{input_dir}/X_test_scaled.npy")
        self.y_train = np.load(f"{input_dir}/y_train.npy")
        self.y_test = np.load(f"{input_dir}/y_test.npy")

        with open(f"{input_dir}/scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)

        print(f"Training data loaded from {input_dir}/")
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test

    def full_pipeline(self, filepath, target_column="visit_per_week",
                     bins=[0, 1, 3, 7], labels=[0, 1, 2],
                     columns_to_drop=["id", "name_personal_trainer", "birthday"],
                     save_data=False, output_dir="data/processed"):
        """Execute the complete data processing pipeline"""
        self.load_data(filepath)
        self.preview_data()
        self.create_target(target_column, bins, labels)
        self.drop_columns(columns_to_drop)
        self.handle_missing_values()
        self.encode_categorical()
        self.train_test_split_data()
        self.normalize_data()

        print("Gotowe do trenowania!")
        self.df
        print("X_train shape:", self.X_train_scaled.shape)

        if save_data:
            self.save_training_data(output_dir)

        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
