from src.processing.proccessing import DataNormalizer
from src.models.gym_model import AbonementPredictor
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt


def create_analysis_graphs():

    df = pd.read_csv("data/raw/gym_membership_realistic(1).csv")

    numeric_cols = ["Age", "visit_per_week", "attend_group_lesson", "drink_abo", "personal_training", "uses_sauna"]


    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    feature_means = df.groupby('abonoment_type')[numeric_cols].mean()
    feature_means.T.plot(kind='bar', ax=axes[0], color=['#1f77b4', '#ff7f0e'], width=0.7)
    axes[0].set_title('Feature x Abonement Type', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Mean Value')
    axes[0].set_xlabel('Features')
    axes[0].legend(title='Abonement Type', loc='upper right')
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].tick_params(axis='x', rotation=45)


    axes[1].hist(df['visit_per_week'], bins=range(0, 9), color='#2ca02c', edgecolor='black', alpha=0.7)
    axes[1].set_title('Distribution of Visits', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Visits per Week')
    axes[1].set_ylabel('Number of Members')
    axes[1].grid(True, alpha=0.3, axis='y')

    axes[2].hist(
        df['personal_training'].astype(int),
        bins=[0, 1, 2],
        color='#ff7f0e',
        edgecolor='black',
        alpha=0.7
    )
    axes[2].set_title('Distribution of Group Lessons', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Attends Group Lesson (0=No, 1=Yes)')
    axes[2].set_ylabel('Number of Members')
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('data/analysis_graphs.png', dpi=300, bbox_inches='tight')


def main():
    create_analysis_graphs()
    normalizer = DataNormalizer()
    X_train, X_test, y_train, y_test, feature_names = normalizer.full_pipeline(
        filepath="data/raw/gym_membership_realistic(1).csv",
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

    model = AbonementPredictor("models/gym_model.pkl")
    model.train(X_train, y_train)
    model.save()

    model.load_preprocessing("data/processed")

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


    print("\nTraining complete. Model saved to models/gym_model.pkl")
    print(model.evaluate(X_test, y_test))

if __name__ == "__main__":
    main()
