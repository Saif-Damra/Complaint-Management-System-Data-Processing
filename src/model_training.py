"""
Model Training Module for Complaint Management System
======================================================
This module handles model training, evaluation, and comparison
for complaint type classification.

Models included:
- K-Nearest Neighbors (KNN)
- Decision Tree
- Gaussian Naive Bayes
- Random Forest
- Support Vector Machine (SVM)
- Logistic Regression
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ──────────────────────────────────────────────
#  Model Definitions
# ──────────────────────────────────────────────

def get_models() -> dict:
    """
    Return a dictionary of models to train and evaluate.
    
    Returns
    -------
    dict
        Model name -> model instance.
    """
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=10),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42, max_depth=15, n_jobs=-1
        ),
        "SVM": SVC(kernel="rbf", random_state=42),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, n_jobs=-1
        ),
    }
    return models


# ──────────────────────────────────────────────
#  Cross-Validation Evaluation
# ──────────────────────────────────────────────

def evaluate_with_cross_validation(
    X, y, n_splits: int = 10, random_state: int = 42
) -> pd.DataFrame:
    """
    Evaluate all models using Stratified K-Fold Cross Validation.
    
    This replaces the manual loop approach with a more robust
    and statistically sound method.
    
    Parameters
    ----------
    X : array-like
        Feature matrix.
    y : array-like
        Target vector.
    n_splits : int
        Number of cross-validation folds.
    random_state : int
        Random state for reproducibility.
    
    Returns
    -------
    pd.DataFrame
        Comparison table with metrics for each model.
    """
    print("\n" + "=" * 60)
    print("🤖 MODEL TRAINING & EVALUATION")
    print(f"   Method: {n_splits}-Fold Stratified Cross Validation")
    print("=" * 60)
    
    # Scale features (important for KNN, SVM, Logistic Regression)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    models = get_models()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    results = {}
    scoring_metrics = ["accuracy", "precision", "recall", "f1"]
    
    for name, model in models.items():
        print(f"\n📌 Training: {name}...")
        
        model_results = {}
        for metric in scoring_metrics:
            scores = cross_val_score(
                model, X_scaled, y, cv=cv, scoring=metric, n_jobs=-1
            )
            model_results[metric] = {
                "mean": scores.mean(),
                "std": scores.std(),
            }
        
        results[name] = model_results
        
        print(f"   Accuracy:  {model_results['accuracy']['mean']:.4f} "
              f"(±{model_results['accuracy']['std']:.4f})")
        print(f"   Precision: {model_results['precision']['mean']:.4f} "
              f"(±{model_results['precision']['std']:.4f})")
        print(f"   Recall:    {model_results['recall']['mean']:.4f} "
              f"(±{model_results['recall']['std']:.4f})")
        print(f"   F1-Score:  {model_results['f1']['mean']:.4f} "
              f"(±{model_results['f1']['std']:.4f})")
    
    # Build comparison DataFrame
    report_data = {}
    for name, metrics in results.items():
        report_data[name] = {
            "Accuracy": f"{metrics['accuracy']['mean']:.4f} (±{metrics['accuracy']['std']:.4f})",
            "Precision": f"{metrics['precision']['mean']:.4f} (±{metrics['precision']['std']:.4f})",
            "Recall": f"{metrics['recall']['mean']:.4f} (±{metrics['recall']['std']:.4f})",
            "F1-Score": f"{metrics['f1']['mean']:.4f} (±{metrics['f1']['std']:.4f})",
        }
    
    report_df = pd.DataFrame(report_data)
    
    # Also build a numeric-only DataFrame for visualization
    numeric_data = {}
    for name, metrics in results.items():
        numeric_data[name] = {
            "Accuracy": metrics["accuracy"]["mean"],
            "Precision": metrics["precision"]["mean"],
            "Recall": metrics["recall"]["mean"],
            "F1-Score": metrics["f1"]["mean"],
        }
    
    numeric_df = pd.DataFrame(numeric_data)
    
    print("\n" + "=" * 60)
    print("📊 MODEL COMPARISON RESULTS")
    print("=" * 60)
    print(report_df.to_string())
    
    # Find best model
    best_model_name = max(
        results.keys(),
        key=lambda k: results[k]["f1"]["mean"]
    )
    best_f1 = results[best_model_name]["f1"]["mean"]
    print(f"\n🏆 Best Model: {best_model_name} (F1-Score: {best_f1:.4f})")
    print("=" * 60)
    
    return report_df, numeric_df, results


# ──────────────────────────────────────────────
#  Detailed Model Evaluation
# ──────────────────────────────────────────────

def detailed_evaluation(X, y, random_state: int = 42):
    """
    Train each model on a single train/test split and produce
    detailed classification reports and confusion matrices.
    
    Parameters
    ----------
    X : array-like
        Feature matrix.
    y : array-like
        Target vector.
    random_state : int
        Random state for reproducibility.
    
    Returns
    -------
    dict
        Model name -> (y_test, y_pred, confusion_matrix, classification_report).
    """
    from sklearn.model_selection import train_test_split
    
    print("\n" + "=" * 60)
    print("🔍 DETAILED MODEL EVALUATION")
    print("=" * 60)
    
    # Scale and split
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=random_state, stratify=y
    )
    
    models = get_models()
    detailed_results = {}
    
    for name, model in models.items():
        print(f"\n📌 {name}:")
        print("-" * 40)
        
        # Train
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Metrics
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        print(f"   Confusion Matrix:\n{cm}")
        print(f"\n   Classification Report:\n{report}")
        
        detailed_results[name] = {
            "y_test": y_test,
            "y_pred": y_pred,
            "confusion_matrix": cm,
            "classification_report": report,
            "model": model,
        }
    
    return detailed_results
