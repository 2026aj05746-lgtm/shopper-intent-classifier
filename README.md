# Online Shopper Purchase-Intent Classification

Machine Learning Assignment 2 – M.Tech (AIML), BITS Pilani WILP

## a. Problem Statement

Predict whether a visitor to an e-commerce website will complete a purchase during a browsing session, using page-visit counts, time spent, bounce/exit rates, visitor type, month and other session attributes. The target is `Revenue` (True = purchase, False = no purchase), so this is a **binary classification** problem. Five classifiers are trained on the same data, compared on six metrics, and exposed through a Streamlit web app.

## b. Dataset Description

- **Source:** UCI Machine Learning Repository – *Online Shoppers Purchasing Intention Dataset*
- **Instances:** 12,330 sessions (one row per session, each from a different user over one year)
- **Features:** 17 input features + 1 target (`Revenue`), no missing values
- **Class balance:** 10,422 no-purchase (84.5%) vs 1,908 purchase (15.5%) – imbalanced
- **Requirement check:** > 500 instances and ≥ 12 features ✔

| Feature group | Columns |
|---|---|
| Page counts & time | `Administrative`, `Administrative_Duration`, `Informational`, `Informational_Duration`, `ProductRelated`, `ProductRelated_Duration` |
| Google Analytics metrics | `BounceRates`, `ExitRates`, `PageValues` |
| Calendar | `SpecialDay` (closeness to a special day), `Month`, `Weekend` |
| Visitor / technical | `VisitorType`, `OperatingSystems`, `Browser`, `Region`, `TrafficType` |

**Preprocessing**
- Stratified 80/20 train/test split (`random_state=42`): 9,864 train / 2,466 test rows.
- Numeric features standardised (not for the Decision Tree).
- `Month`, `VisitorType`, `OperatingSystems`, `Browser`, `Region`, `TrafficType` one-hot encoded (not for Gaussian Naive Bayes, which expects continuous inputs).
- Preprocessing and classifier are saved together as one scikit-learn `Pipeline` per model, so the app can score a raw CSV directly.
- `class_weight="balanced"` is used for Logistic Regression, Decision Tree and SVM to handle the imbalance.

## c. GitHub Repository Link

**<PASTE YOUR GITHUB REPO URL HERE>**

Live app: **<PASTE YOUR STREAMLIT APP URL HERE>**

```
├── app.py                    Streamlit app
├── train_models.py           trains & saves the 5 models, writes test_data.csv
├── online_shoppers_intention.csv   full dataset
├── test_data.csv             held-out test split (2,466 rows) for the app
├── requirements.txt
├── metrics_summary.csv
└── model/                    5 saved pipelines (.joblib)
```

Run locally: `pip install -r requirements.txt`, then `python train_models.py` (optional) and `streamlit run app.py`.

## d. Models Used

All metrics are computed on the same 2,466-row hold-out test set (positive class = purchase).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8410 | 0.8933 | 0.4913 | 0.7435 | 0.5917 | 0.5145 |
| Decision Tree | 0.8508 | 0.9137 | 0.5113 | 0.8298 | 0.6327 | 0.5708 |
| KNN | 0.8735 | 0.8149 | 0.6944 | 0.3272 | 0.4448 | 0.4184 |
| Naive Bayes | 0.8058 | 0.8014 | 0.4171 | 0.6387 | 0.5047 | 0.4041 |
| Support Vector Machine | 0.8658 | 0.8953 | 0.5491 | 0.7461 | 0.6326 | 0.5625 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Solid linear baseline with AUC 0.893 and recall 0.74 thanks to class weighting. Precision is low (0.49): about half of the predicted buyers are not buyers, which pulls accuracy down to 0.84. |
| Decision Tree | Best AUC (0.914), best recall (0.83) and best MCC (0.571). A shallow tree (depth 6) picks up the strong `PageValues` effect and non-linear interactions without overfitting. Precision is moderate (0.51). |
| KNN | Highest accuracy (0.874) but this is misleading: recall is only 0.33, so it misses two-thirds of buyers by mostly predicting the majority class. It has the lowest F1 (0.44) and weak AUC (0.815), and it is also the largest saved model. |
| Naive Bayes | Weakest overall (MCC 0.40, AUC 0.80). The independence assumption is violated because page counts and durations are strongly correlated, and the Gaussian assumption fits skewed count data poorly. It is very fast, though, and has reasonable recall (0.64). |
| Support Vector Machine | Best balance after the tree: accuracy 0.866, F1 0.633 (practically tied with the tree) and MCC 0.563. The RBF kernel captures non-linear structure well, but training and probability calibration are slower than for the other models. |
| **Overall winner** | **Decision Tree** – highest AUC, recall and MCC and joint-best F1, while being tiny and interpretable. The SVM is a close runner-up. Accuracy alone would wrongly favour KNN because of the 85/15 class imbalance, so MCC/F1/AUC were used for the decision. |

## Streamlit App Features
1. CSV upload (test data with all 17 features and `Revenue`)
2. Model selection dropdown (5 models)
3. Accuracy, AUC, Precision, Recall, F1 and MCC displayed
4. Confusion matrix, ROC curve and classification report
5. Optional side-by-side comparison table of all models

The app loads the saved models and only predicts; nothing is retrained on the uploaded data.
