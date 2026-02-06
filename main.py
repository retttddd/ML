from src.processing.proccessing import DataNormalizer
from src.models.gym_model import AbonementPredictorModel

if __name__ == '__main__':

    normalizer = DataNormalizer()
    X_train, X_test, y_train, y_test, feature_names = normalizer.full_pipeline(
        filepath="data/raw/gym_membership.csv",
        target_column="abonoment_type",
        columns_to_drop=["id", "avg_time_check_out", "avg_time_in_gym", "name_personal_trainer", "birthday", "days_per_week", "visit_per_week", "fav_group_lesson", "fav_drink", ],
        save_data=True,
        output_dir="data/processed"
    )

    model = AbonementPredictorModel(model_path="models/gym_model.pkl")
    model.train(X_train, y_train, n_estimators=100, random_state=42)
    model.save_model()

    model.load_preprocessing_artifacts("data/processed")
    ##accuracy = model.evaluate(X_test, y_test)

    print(f"\nFeature names: {feature_names}")

    print("\n" + "=" * 60)
    print("EXAMPLE: Predict abonement for a new user")
    print("=" * 60)

    example_user = {
        'Age': 30,
        'gender_Male': 1,  # 1=Male, 0=Female
        'attend_group_lesson': 3,
        'avg_time_in_gym': 90,
        'drink_abo': 1,
        'personal_training': 0,
        'uses_sauna': 0
    }

    result = model.predict_abonement(example_user)
    print(f"\nUser data: {example_user}")
    print(f"\nRecommended Abonement: {result['abonement_type']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"\nAll probabilities:")
    for abo_type, prob in result['probabilities'].items():
        print(f"  {abo_type}: {prob:.2%}")

    print("\n" + "=" * 60)
    print("Training Complete! Model saved to models/gym_model.pkl")
    print("=" * 60)
