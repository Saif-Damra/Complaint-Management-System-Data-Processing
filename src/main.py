"""
Main Entry Point - Complaint Management System Data Processing
===============================================================
This script orchestrates the complete pipeline:
1. Data Loading & Cleaning
2. Exploratory Data Analysis (EDA) Visualizations
3. Model Training & Evaluation (with Cross Validation)
4. Detailed Model Evaluation (Confusion Matrices)
5. Model Comparison Visualizations

Usage:
    python src/main.py
    python src/main.py --data path/to/Complaints.csv
    python src/main.py --output results/
"""

import argparse
import sys
import os
import warnings

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing import load_data, analyze_missing_values, \
    drop_high_missing_columns, drop_leaky_columns, fill_missing_values, \
    add_time_features, encode_categorical, process_data
from src.model_training import evaluate_with_cross_validation, detailed_evaluation
from src.visualization import run_eda_visualizations, run_model_visualizations


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Complaint Management System - Data Processing & ML Pipeline"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/Complaints.csv",
        help="Path to the complaints CSV file (default: data/Complaints.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs",
        help="Directory to save output files (default: outputs/)",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=10,
        help="Number of cross-validation folds (default: 10)",
    )
    parser.add_argument(
        "--skip-eda",
        action="store_true",
        help="Skip EDA visualizations",
    )
    return parser.parse_args()


def main():
    """Run the complete pipeline."""
    args = parse_args()
    
    # Check data file exists
    if not os.path.exists(args.data):
        # Try alternate paths
        alternate_paths = [
            "data/Complaints.csv",
            "Complaints.csv",
            "Notebook/Complaints.csv",
        ]
        found = False
        for alt_path in alternate_paths:
            if os.path.exists(alt_path):
                args.data = alt_path
                found = True
                break
        
        if not found:
            print(f"❌ Error: Data file not found at '{args.data}'")
            print(f"   Please place 'Complaints.csv' in the data/ directory")
            print(f"   Or specify the path: python src/main.py --data path/to/file.csv")
            sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print("╔" + "═" * 58 + "╗")
    print("║   COMPLAINT MANAGEMENT SYSTEM - DATA PROCESSING PIPELINE  ║")
    print("╚" + "═" * 58 + "╝")
    print()
    
    # ── Step 1: Load raw data for EDA ──
    raw_df = load_data(args.data)
    
    # ── Step 2: EDA Visualizations (on raw data) ──
    if not args.skip_eda:
        run_eda_visualizations(raw_df, args.output)
    
    # ── Step 3: Process data ──
    X, y, encoders = process_data(args.data)
    
    # ── Step 4: Model Training with Cross Validation ──
    report_df, numeric_df, cv_results = evaluate_with_cross_validation(
        X, y, n_splits=args.folds
    )
    
    # ── Step 5: Detailed Evaluation ──
    detailed_results = detailed_evaluation(X, y)
    
    # ── Step 6: Model Comparison Visualizations ──
    run_model_visualizations(numeric_df, detailed_results, args.output)
    
    # ── Step 7: Save results to CSV ──
    results_path = os.path.join(args.output, "model_results.csv")
    report_df.to_csv(results_path)
    print(f"\n📁 Results saved to: {results_path}")
    
    # ── Summary ──
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║                    PIPELINE COMPLETE!                     ║")
    print("╚" + "═" * 58 + "╝")
    print(f"\n📂 All outputs saved to: {args.output}/")
    print("   ├── complaint_type_distribution.png")
    print("   ├── escalation_analysis.png")
    print("   ├── actual_complaint_distribution.png")
    print("   ├── callback_mechanism.png")
    print("   ├── customer_type_distribution.png")
    print("   ├── model_comparison.png")
    print("   ├── model_radar_chart.png")
    print("   ├── confusion_matrices.png")
    print("   ├── performance_heatmap.png")
    print("   └── model_results.csv")


if __name__ == "__main__":
    main()
