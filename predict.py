from src.processing.proccessing import DataNormalizer
from src.models.gym_model import AbonementPredictorModel

def predict_abonement_for_user():
    print("=" * 60)
    print("GYM ABONEMENT PREDICTOR")
    print("=" * 60)

    model = AbonementPredictorModel(model_path="models/gym_model.pkl")
    model.load_model()
    model.load_preprocessing_artifacts("data/processed")
    
    print("\nEnter user information:")
    print("-" * 60)

    try:
        visit_per_week = int(input("Visits per week (e.g., 1-7): "))
        num_days_per_week = int(input("Number of different days per week (e.g., 1-7): "))
        age = int(input("Age: "))
        attend_group_lesson = int(input("Attends group lessons? (1=Yes, 0=No): "))
        avg_time_in_gym = int(input("Average time in gym (minutes): "))
        drink_abo = int(input("Has drink subscription? (1=Yes, 0=No): "))
        personal_training = int(input("Has personal training? (1=Yes, 0=No): "))
        uses_sauna = int(input("Uses sauna? (1=Yes, 0=No): "))

        user_data = {
            'visit_per_week': visit_per_week,
            'num_days_per_week': num_days_per_week,
            'Age': age,
            'gender_Male': 1,
            'attend_group_lesson': attend_group_lesson,
            'avg_time_in_gym': avg_time_in_gym,
            'drink_abo': drink_abo,
            'personal_training': personal_training,
            'uses_sauna': uses_sauna
        }

        result = model.predict_abonement(user_data)
        
        print("\n" + "=" * 60)
        print("PREDICTION RESULT")
        print("=" * 60)
        print(f"\nRecommended Abonement: {result['abonement_type']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"\nProbabilities for all abonement types:")
        for abo_type, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {abo_type}: {prob:.2%}")
        print("=" * 60)
        
    except ValueError as e:
        print(f"\nError: Invalid input. Please enter valid numbers.")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == '__main__':
    predict_abonement_for_user()
