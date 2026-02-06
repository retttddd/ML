from src.processing.proccessing import DataNormalizer
from src.models.gym_model import AbonementPredictor


def main():
    # ---------- Data preprocessing ----------
    normalizer = DataNormalizer()

    X_train, X_test, y_train, y_test, feature_names = normalizer.full_pipeline(
        filepath="data/raw/gym_membership.csv",
        target_column="abonoment_type",
        columns_to_drop=[
            "id",
            "name_personal_trainer",
            "birthday",
            "days_per_week",
            "avg_time_check_in",
            "avg_time_check_out",
            "fav_group_lesson",
            "fav_drink",
        ],
        save_data=True,
        output_dir="data/processed"
    )

    print(f"Features used: {feature_names}")

    # ---------- Model training ----------
    model = AbonementPredictor("models/gym_model.pkl")
    model.train(X_train, y_train)
    model.save()

    # ---------- Load preprocessing----------
    model.load_preprocessing("data/processed")

    # ---------- Example prediction ----------
    example_user = {
        "Age": 30,
        "gender": "Male",
        "attend_group_lesson": 1,
        "visit_per_week": 3,
        "avg_time_in_gym": 90,
        "drink_abo": 1,
        "personal_training": 0,
        "uses_sauna": 0,
    }

    result = model.predict(example_user)


    print("\nPrediction result")
    print("Recommended abonement:", result["abonement_type"])
    print(f"Confidence: {result['confidence']:.2%}")

    # print("\nProbabilities:")
    # for name, prob in result["probabilities"].items():
    #     print(f"  {name}: {prob:.2%}")

    print("\nTraining complete. Model saved to models/gym_model.pkl")
    print(model.evaluate(X_test, y_test))

if __name__ == "__main__":
    main()
