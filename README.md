# Cricket Wizard - IPL Fantasy Team Predictor

---

## Project Description

This project is a machine learning model that predicts the best fantasy IPL team based on historical player performance data. The model utilizes past IPL scores and other relevant statistics from merged datasets to make predictions.

The IPL Fantasy Team Predictor leverages historical Indian Premier League (IPL) data to forecast player performance and assist in selecting optimal fantasy cricket teams. By combining multiple datasets and applying advanced data engineering techniques, the model generates aggregate statistics and features to improve prediction accuracy. The system analyzes factors such as player form, match conditions, and historical performance to recommend the most promising players for fantasy team selection.

---

## Model

The machine learning model uses **XGBoost** (Extreme Gradient Boosting), a powerful and efficient implementation of gradient boosted decision trees.

Key aspects of the model include:

- Feature engineering to create advanced metrics like batting impact, bowling impact, and overall performance.
- Cross-validation for model evaluation.
- Hyperparameter tuning for optimal performance.
- Rolling window prediction for 2023 matches.

The model achieves high accuracy by utilizing various performance metrics:

- Binary Cross-Entropy Loss
- Accuracy
- ROC-AUC Score

These metrics are calculated for each match prediction and overall model performance, ensuring robust evaluation of the model's predictive capabilities.

---

## Dataset

The dataset used for this project is a result of merging multiple IPL-related datasets and includes:

- Match details (venue, teams, date, etc.)
- Player statistics (runs, wickets, economy rate, strike rate, etc.)
- Derived features (moving averages, form indicators, etc.)
- Fantasy points scored in previous matches

---

## Data Engineering

Several data engineering steps were performed to prepare the dataset:

- Merging of multiple source datasets.
- Feature engineering to create aggregate statistics.
- Calculation of moving averages and form indicators.
- Normalization of player statistics across seasons.

---

## Usage

With the correct dataset path in the main code file, run the main code file.

---

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- XGBoost

---

## Installation

1. Install the necessary libraries and dependencies.
2. Download the dataset.
3. In the main code file, fill the path with your relative dataset path.
4. Run the code.

---

## Contributing

Feel free to improvise and contribute to this project. Thank you!

---

## License

---

## Acknowledgments

- IPL for providing the original data.
