"""
Visualization Module for Complaint Management System
=====================================================
This module creates professional, publication-quality visualizations
for complaint data analysis and model comparison.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

# ──────────────────────────────────────────────
#  Style Configuration
# ──────────────────────────────────────────────

# Use a clean, modern style
plt.style.use("seaborn-v0_8-whitegrid")
matplotlib.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 16,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

# Custom color palette
COLORS = {
    "primary": "#2563EB",
    "secondary": "#7C3AED",
    "success": "#059669",
    "warning": "#D97706",
    "danger": "#DC2626",
    "info": "#0891B2",
}

MODEL_COLORS = ["#2563EB", "#7C3AED", "#059669", "#D97706", "#DC2626", "#0891B2"]


def ensure_output_dir(output_dir: str = "outputs"):
    """Create output directory if it doesn't exist."""
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


# ──────────────────────────────────────────────
#  EDA Visualizations
# ──────────────────────────────────────────────

def plot_complaint_type_distribution(df: pd.DataFrame, output_dir: str = "outputs"):
    """
    Plot the distribution of complaint types (target variable).
    """
    ensure_output_dir(output_dir)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Complaint Type Distribution", fontweight="bold", fontsize=16)
    
    # Bar chart
    counts = df["COMPLAINT_TYPE"].value_counts()
    colors = [COLORS["primary"], COLORS["secondary"]]
    bars = axes[0].bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=2)
    axes[0].set_title("Count by Type")
    axes[0].set_ylabel("Number of Complaints")
    
    # Add value labels on bars
    for bar, val in zip(bars, counts.values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
            str(val), ha="center", va="bottom", fontweight="bold", fontsize=12
        )
    
    # Pie chart
    axes[1].pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        textprops={"fontsize": 12},
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        explode=[0.02] * len(counts),
    )
    axes[1].set_title("Proportion by Type")
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, "complaint_type_distribution.png")
    plt.savefig(filepath)
    plt.close()
    print(f"   📈 Saved: {filepath}")


def plot_escalation_analysis(df: pd.DataFrame, output_dir: str = "outputs"):
    """
    Plot escalation flag distribution and its relationship with complaint type.
    """
    ensure_output_dir(output_dir)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Escalation Analysis", fontweight="bold", fontsize=16)
    
    # Escalation flag distribution
    esc_counts = df["ESCALATION_FLAG"].value_counts()
    colors = [COLORS["success"], COLORS["danger"]]
    axes[0].bar(esc_counts.index, esc_counts.values, color=colors, edgecolor="white", linewidth=2)
    axes[0].set_title("Escalation Flag Distribution")
    axes[0].set_ylabel("Count")
    
    for i, (idx, val) in enumerate(zip(esc_counts.index, esc_counts.values)):
        axes[0].text(i, val + 50, str(val), ha="center", fontweight="bold")
    
    # Escalation by complaint type
    ct = pd.crosstab(df["COMPLAINT_TYPE"], df["ESCALATION_FLAG"])
    ct.plot(
        kind="bar",
        ax=axes[1],
        color=[COLORS["success"], COLORS["danger"]],
        edgecolor="white",
        linewidth=2,
    )
    axes[1].set_title("Escalation by Complaint Type")
    axes[1].set_ylabel("Count")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
    axes[1].legend(title="Escalated")
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, "escalation_analysis.png")
    plt.savefig(filepath)
    plt.close()
    print(f"   📈 Saved: {filepath}")


def plot_actual_complaint_distribution(df: pd.DataFrame, output_dir: str = "outputs"):
    """
    Plot the distribution of actual vs non-actual complaints.
    """
    ensure_output_dir(output_dir)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    counts = df["ACTUAL_COMPLAINT"].value_counts()
    colors = [COLORS["primary"], COLORS["warning"], COLORS["secondary"]][:len(counts)]
    
    bars = ax.barh(counts.index, counts.values, color=colors, edgecolor="white", linewidth=2)
    ax.set_title("Actual Complaint Distribution", fontweight="bold", fontsize=16)
    ax.set_xlabel("Count")
    
    for bar, val in zip(bars, counts.values):
        ax.text(
            val + 20, bar.get_y() + bar.get_height() / 2,
            str(val), va="center", fontweight="bold"
        )
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, "actual_complaint_distribution.png")
    plt.savefig(filepath)
    plt.close()
    print(f"   📈 Saved: {filepath}")


def plot_callback_mechanism(df: pd.DataFrame, output_dir: str = "outputs"):
    """
    Plot callback mechanism distribution.
    """
    ensure_output_dir(output_dir)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    counts = df["CALLBACK_MECHANISM"].value_counts()
    colors = sns.color_palette("viridis", len(counts))
    
    ax.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        textprops={"fontsize": 11},
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    ax.set_title("Callback Mechanism Distribution", fontweight="bold", fontsize=16)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, "callback_mechanism.png")
    plt.savefig(filepath)
    plt.close()
    print(f"   📈 Saved: {filepath}")


def plot_customer_type_distribution(df: pd.DataFrame, output_dir: str = "outputs"):
    """
    Plot customer type distribution.
    """
    ensure_output_dir(output_dir)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    counts = df["CUSTOMER_TYPE"].value_counts()
    colors = sns.color_palette("coolwarm", len(counts))
    
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=2)
    ax.set_title("Customer Type Distribution", fontweight="bold", fontsize=16)
    ax.set_ylabel("Count")
    ax.set_xlabel("Customer Type")
    
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
            str(val), ha="center", fontweight="bold"
        )
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, "customer_type_distribution.png")
    plt.savefig(filepath)
    plt.close()
    print(f"   📈 Saved: {filepath}")


# ──────────────────────────────────────────────
#  Model Comparison Visualizations
# ──────────────────────────────────────────────

def plot_model_comparison(numeric_df: pd.DataFrame, output_dir: str = "outputs"):
    """
    Create a comprehensive model comparison visualization.
    
    Parameters
    ----------
    numeric_df : pd.DataFrame
        DataFrame with numeric metric values (index=metrics, columns=models).
    """
    ensure_output_dir(output_dir)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Model Comparison Dashboard", fontweight="bold", fontsize=18, y=1.02)
    
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    
    for idx, (ax, metric) in enumerate(zip(axes.flat, metrics)):
        values = numeric_df.loc[metric]
        models = values.index.tolist()
        scores = values.values
        
        bars = ax.bar(
            models, scores,
            color=MODEL_COLORS[:len(models)],
            edgecolor="white",
            linewidth=2,
        )
        
        ax.set_title(metric, fontweight="bold", fontsize=14)
        ax.set_ylim(0, 1.1)
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=9)
        
        # Add value labels
        for bar, score in zip(bars, scores):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{score:.3f}",
                ha="center", va="bottom", fontweight="bold", fontsize=10,
            )
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(filepath)
    plt.close()
    print(f"   📈 Saved: {filepath}")


def plot_model_radar(numeric_df: pd.DataFrame, output_dir: str = "outputs"):
    """
    Create a radar/spider chart comparing all models across metrics.
    """
    ensure_output_dir(output_dir)
    
    metrics = numeric_df.index.tolist()
    models = numeric_df.columns.tolist()
    
    # Number of variables
    N = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    for i, model in enumerate(models):
        values = numeric_df[model].values.tolist()
        values += values[:1]  # Close the polygon
        
        ax.plot(angles, values, "o-", linewidth=2,
                label=model, color=MODEL_COLORS[i % len(MODEL_COLORS)])
        ax.fill(angles, values, alpha=0.1,
                color=MODEL_COLORS[i % len(MODEL_COLORS)])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Performance Radar Chart",
                 fontweight="bold", fontsize=16, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, "model_radar_chart.png")
    plt.savefig(filepath)
    plt.close()
    print(f"   📈 Saved: {filepath}")


def plot_confusion_matrices(detailed_results: dict, output_dir: str = "outputs"):
    """
    Plot confusion matrices for all models in a grid.
    
    Parameters
    ----------
    detailed_results : dict
        Output from model_training.detailed_evaluation().
    """
    ensure_output_dir(output_dir)
    
    n_models = len(detailed_results)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    fig.suptitle("Confusion Matrices", fontweight="bold", fontsize=18, y=1.02)
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (name, result) in enumerate(detailed_results.items()):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        
        cm = result["confusion_matrix"]
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            ax=ax, cbar=False,
            annot_kws={"size": 14, "fontweight": "bold"},
            linewidths=2, linecolor="white",
        )
        ax.set_title(name, fontweight="bold", fontsize=13)
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
    
    # Hide empty subplots
    for idx in range(n_models, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, "confusion_matrices.png")
    plt.savefig(filepath)
    plt.close()
    print(f"   📈 Saved: {filepath}")


def plot_heatmap_comparison(numeric_df: pd.DataFrame, output_dir: str = "outputs"):
    """
    Create a heatmap showing all metrics for all models.
    """
    ensure_output_dir(output_dir)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.heatmap(
        numeric_df.astype(float),
        annot=True, fmt=".4f",
        cmap="YlGnBu",
        ax=ax,
        linewidths=2, linecolor="white",
        annot_kws={"size": 13, "fontweight": "bold"},
        vmin=0.5, vmax=1.0,
    )
    
    ax.set_title("Model Performance Heatmap", fontweight="bold", fontsize=16)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=12)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, "performance_heatmap.png")
    plt.savefig(filepath)
    plt.close()
    print(f"   📈 Saved: {filepath}")


# ──────────────────────────────────────────────
#  Run All Visualizations
# ──────────────────────────────────────────────

def run_eda_visualizations(df: pd.DataFrame, output_dir: str = "outputs"):
    """
    Run all EDA visualizations on the raw (pre-encoded) data.
    """
    print("\n" + "=" * 60)
    print("📊 EXPLORATORY DATA ANALYSIS VISUALIZATIONS")
    print("=" * 60)
    
    plot_complaint_type_distribution(df, output_dir)
    plot_escalation_analysis(df, output_dir)
    plot_actual_complaint_distribution(df, output_dir)
    plot_callback_mechanism(df, output_dir)
    plot_customer_type_distribution(df, output_dir)
    
    print(f"\n✅ All EDA visualizations saved to '{output_dir}/'")


def run_model_visualizations(
    numeric_df: pd.DataFrame,
    detailed_results: dict,
    output_dir: str = "outputs",
):
    """
    Run all model comparison visualizations.
    """
    print("\n" + "=" * 60)
    print("📊 MODEL COMPARISON VISUALIZATIONS")
    print("=" * 60)
    
    plot_model_comparison(numeric_df, output_dir)
    plot_model_radar(numeric_df, output_dir)
    plot_confusion_matrices(detailed_results, output_dir)
    plot_heatmap_comparison(numeric_df, output_dir)
    
    print(f"\n✅ All model visualizations saved to '{output_dir}/'")
