- [x] Data processing and normalizing
- [x] Saving trained data
- [x] Model training (Random Forest)
- [x] Training pipeline
- [x] Prediction script
- [ ] API endpoint

├── data/
│   ├── processed/  (training artifacts)
│   └── raw/        (gym_membership.csv)
├── models/         (gym_model.pkl)
├── src/
│   ├── models/     (gym_model.py)
│   └── processing/ (proccessing.py)
├── main.py         (training pipeline)
├── predict.py      (prediction script)
└── README.md 