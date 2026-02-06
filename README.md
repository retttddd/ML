## ML projekt zaliczeniowy
Ivan  Stsepaniuk, ...,  .... \
System predykcji typu abonamentu dla klientów siłowni. Model ML (76% accuracy) analizuje dane
użytkowników (wiek, częstotliwość wizyt, czas w siłowni, korzystanie z trenera personalnego, sauny itp.)
i rekomenduje odpowiedni abonament (Standard lub Premium)

## TODO: 
- [x] Data processing and normalizing
- [x] Saving trained data
- [x] Model training (Random Forest)
- [x] Training pipeline
- [x] Prediction script
- [ ] API endpoint

## Graphs
    Przykłady wag poszczególnych cech oraz ich dystrybucji.
![analysis_graphs.png](data/analysis_graphs.png)

## Results
```
Distribution:
abonoment_type
Standard    590
Premium     410

                visit_per_week  avg_time_in_gym  ...  drink_abo  uses_sauna
abonoment_type                                   ...                       
Premium               3.760976        90.253659  ...   0.431707    0.536585
Standard              2.515254        68.376271  ...   0.269492    0.288136

[2 rows x 6 columns]
     Age  abonoment_type  ...  personal_training_True  uses_sauna_True
0     41               1  ...                   False            False
1     33               1  ...                   False            False
2     43               1  ...                   False            False
3     53               1  ...                    True            False
4     32               1  ...                   False            False
..   ...             ...  ...                     ...              ...
995   32               0  ...                    True            False
996   57               1  ...                   False            False
997   43               1  ...                   False            False
998   28               0  ...                   False            False
999   42               1  ...                   False            False

[1000 rows x 13 columns] used

Feature count: 12
Features used: ['Age', 'visit_per_week', 'avg_time_in_gym', 'gender_Male', 'attend_group_lesson_True', 'Age_bucket_18_29', 'Age_bucket_30_44', 'Age_bucket_45_59', 'Age_bucket_60_plus', 'drink_abo_True', 'personal_training_True', 'uses_sauna_True']

Prediction result
Recommended abonement: Standard
Confidence: 74.89%

*** Accuracy: 0.77 ***
              precision    recall  f1-score   support

     Premium       0.77      0.62      0.69        82
    Standard       0.77      0.87      0.82       118

    accuracy                           0.77       200
   macro avg       0.77      0.75      0.75       200
weighted avg       0.77      0.77      0.76       200

```
