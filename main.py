from src.processing.proccessing import DataNormalizer

if __name__ == '__main__':
    normalizer = DataNormalizer()

    X_train, X_test, y_train, y_test = normalizer.full_pipeline(
        filepath="data/raw/gym_membership.csv",
        target_column="visit_per_week",
        bins=[0, 1, 3, 7],
        labels=[0, 1, 2],
        columns_to_drop=["id", "name_personal_trainer", "birthday"],
        save_data=True,
        output_dir="data/processed"
    )
