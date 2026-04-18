"""
Complete Machine Learning Model Implementation Project

- Uses: pandas, numpy, scikit-learn, matplotlib, seaborn
- Dataset: Breast Cancer dataset from scikit-learn
- Steps: data loading, preprocessing, train/test split, model training,
         evaluation, and visualization
"""

# ===========================
# 1. Import required libraries
# ===========================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    roc_curve,
    auc,
)


# ===========================
# 2. Data loading
# ===========================

def load_dataset():
    """
    Load a sample dataset and return it as a pandas DataFrame.

    Here we use the Breast Cancer dataset from scikit-learn.
    We convert it into a DataFrame so we can work with it like a CSV.
    """
    cancer = load_breast_cancer()
    X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    y = pd.Series(cancer.target, name="target")  # 0 = malignant, 1 = benign
    return X, y, cancer


# ===========================
# 3. Preprocessing and train/test split
# ===========================

def preprocess_and_split_data(X, y, test_size=0.2, random_state=42):
    """
    Handle missing values, scale features, and split into train and test sets.

    - Missing values: handled with SimpleImputer (median strategy).
      (The chosen dataset has no missing values, but this step demonstrates how
       to handle them if they exist.)
    - Feature scaling: StandardScaler to give features zero mean and unit variance.
    - Train/test split: stratified to keep class balance.
    """
    # If your real-world dataset has missing values, this imputer will fill them.
    imputer = SimpleImputer(strategy="median")

    # Standardize numerical features.
    scaler = StandardScaler()

    # We will build the full preprocessing inside a Pipeline later for the model,
    # but for splitting we just separate X and y here.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,   # keep class distribution similar in train and test
    )

    return X_train, X_test, y_train, y_test, imputer, scaler


# ===========================
# 4. Model training
# ===========================

def train_model(X_train, y_train, imputer, scaler):
    """
    Train a machine learning model using a Pipeline.

    - Steps:
        1. Impute missing values.
        2. Scale features.
        3. Train a Random Forest classifier.
    """
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    # Build a pipeline that combines preprocessing and model in one object.
    pipeline = Pipeline(
        steps=[
            ("imputer", imputer),
            ("scaler", scaler),
            ("classifier", model),
        ]
    )

    # Fit the pipeline on the training data.
    pipeline.fit(X_train, y_train)

    return pipeline


# ===========================
# 5. Model evaluation
# ===========================

def evaluate_model(model, X_test, y_test, target_names):
    """
    Evaluate the model using accuracy, precision, recall, f1-score,
    confusion matrix, and classification report.

    - Accuracy: overall correct predictions ratio.
    - Confusion matrix: counts of true vs predicted classes.
    - Classification report: precision, recall, f1-score per class.
    """
    # Predict labels for the test set.
    y_pred = model.predict(X_test)

    # Predict probabilities for the positive class (needed for ROC curve).
    # We assume class "1" is the positive class (benign in this dataset).
    y_prob = model.predict_proba(X_test)[:, 1]

    # Compute accuracy.
    accuracy = accuracy_score(y_test, y_pred)

    # Compute confusion matrix.
    cm = confusion_matrix(y_test, y_pred)

    # Compute precision, recall and f1-score (macro average over classes).
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )

    # Generate a detailed classification report.
    report = classification_report(y_test, y_pred, target_names=target_names)

    print("=== Evaluation Metrics ===")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}\n")
    print("Classification Report:")
    print(report)

    return accuracy, cm, y_pred, y_prob


# ===========================
# 6. Visualization
# ===========================

def visualize_results(cm, y_test, y_pred, y_prob, model, feature_names):
    """
    Visualize evaluation results using graphs.

    - Confusion matrix heatmap.
    - Distribution of predicted classes.
    - ROC curve with AUC.
    - Feature importance bar plot (from Random Forest).
    """
    # Set a nice style for plots.
    sns.set(style="whitegrid", context="talk")

    # 6.1 Confusion matrix heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted 0 (malignant)", "Predicted 1 (benign)"],
        yticklabels=["Actual 0 (malignant)", "Actual 1 (benign)"],
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()

    # 6.2 ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random guess")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()

    # 6.3 Bar plot of predicted class counts
    plt.figure(figsize=(6, 4))
    sns.countplot(
        x=y_pred,
        palette="viridis",
    )
    plt.title("Prediction Distribution on Test Set")
    plt.xlabel("Predicted class (0 = malignant, 1 = benign)")
    plt.ylabel("Count")
    plt.tight_layout()

    # 6.4 Feature importances from the Random Forest model
    # Extract the underlying RandomForestClassifier from the pipeline.
    rf_model = model.named_steps["classifier"]
    importances = rf_model.feature_importances_

    # Create a DataFrame for easy plotting of feature importances.
    fi_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values(by="importance", ascending=False)

    # Plot top 10 important features.
    top_n = 10
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=fi_df.head(top_n),
        x="importance",
        y="feature",
        palette="magma",
    )
    plt.title(f"Top {top_n} Feature Importances (Random Forest)")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()

    # Show all plots.
    plt.show()


# ===========================
# 7. Main execution function
# ===========================

def main():
    """
    Main function to run the complete ML workflow:
    - Load data
    - Preprocess and split
    - Train model
    - Evaluate
    - Visualize
    """
    # 7.1 Load the dataset
    X, y, cancer = load_dataset()
    print("Dataset loaded.")
    print(f"Number of samples: {X.shape[0]}, Number of features: {X.shape[1]}")

    # 7.2 Preprocess and split into train/test
    X_train, X_test, y_train, y_test, imputer, scaler = preprocess_and_split_data(X, y)
    print("Data preprocessed and split into train/test sets.")
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # 7.3 Train the model
    model = train_model(X_train, y_train, imputer, scaler)
    print("Model training completed.")

    # 7.4 Evaluate the model
    accuracy, cm, y_pred, y_prob = evaluate_model(model, X_test, y_test, cancer.target_names)
    print(f"Final test accuracy: {accuracy:.4f}")

    # 7.5 Visualize results (including ROC curve)
    visualize_results(cm, y_test, y_pred, y_prob, model, cancer.feature_names)


# ===========================
# 8. Entry point
# ===========================

if __name__ == "__main__":
    # Running this file directly will execute the complete ML pipeline.
    main()

