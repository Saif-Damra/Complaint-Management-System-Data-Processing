"""
Data Processing Module for Complaint Management System
=======================================================
This module handles data loading, cleaning, and preprocessing
for the telecom complaint dataset.
"""

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the complaints dataset from a CSV file.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    
    Returns
    -------
    pd.DataFrame
        Raw complaint data.
    """
    df = pd.read_csv(filepath)
    print(f"✅ Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def analyze_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze and display the percentage of missing values per column.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with missing value statistics.
    """
    missing_count = df.isnull().sum()
    missing_percent = (missing_count * 100) / len(df)
    
    missing_stats = pd.DataFrame({
        "Missing Count": missing_count,
        "Missing %": missing_percent.round(2)
    })
    missing_stats = missing_stats[missing_stats["Missing Count"] > 0]
    missing_stats = missing_stats.sort_values("Missing %", ascending=False)
    
    print("\n📊 Missing Values Analysis:")
    print("=" * 50)
    if len(missing_stats) > 0:
        print(missing_stats.to_string())
    else:
        print("No missing values found!")
    print("=" * 50)
    
    return missing_stats


def drop_high_missing_columns(df: pd.DataFrame, threshold: float = 85.0) -> pd.DataFrame:
    """
    Drop columns with missing value percentage above the threshold.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    threshold : float
        Percentage threshold above which columns are dropped.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with high-missing columns removed.
    """
    high_missing_cols = []
    for col in df.columns:
        missing_pct = df[col].isnull().sum() * 100 / len(df)
        if missing_pct > threshold:
            high_missing_cols.append(col)
    
    if high_missing_cols:
        print(f"\n🗑️  Dropping columns with >{threshold}% missing values: {high_missing_cols}")
        df = df.drop(columns=high_missing_cols)
    
    return df


def drop_leaky_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that cause data leakage for COMPLAINT_TYPE prediction.
    
    The columns 'CASE' and 'PRODUCT' are directly derived from 
    COMPLAINT_TYPE, making them leaky features that would artificially 
    inflate model performance.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with leaky columns removed.
    """
    leaky_cols = [col for col in ["CASE", "PRODUCT"] if col in df.columns]
    
    if leaky_cols:
        print(f"\n⚠️  Dropping data-leaky columns: {leaky_cols}")
        print("   (These columns are directly derived from COMPLAINT_TYPE)")
        df = df.drop(columns=leaky_cols)
    
    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values with appropriate strategies.
    
    Strategy:
    - CLOSE_USER: Fill with OPEN_USER (assumption: same user closes)
    - OPEN_USER: Fill remaining with 'Unknown'
    - CLOSE_DATE: Drop rows (can't impute dates meaningfully)
    - ESCALATED_GROUP: Fill with 'NO_GROUP' (not escalated)
    - CALLBACK_MECHANISM: Fill with mode (most frequent value)
    - OFFER_NAME/CUSTOMER_GROUP: Context-based imputation
    - OPEN_GR/CLOSE_GROUP: Cross-reference imputation
    - AGE_BRACKET: Fill with median
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with missing values handled.
    """
    print("\n🔧 Filling missing values...")
    initial_rows = len(df)
    
    # --- CLOSE_USER & OPEN_USER ---
    df["CLOSE_USER"] = df["CLOSE_USER"].fillna(df["OPEN_USER"])
    df["CLOSE_USER"] = df["CLOSE_USER"].fillna("Unknown")
    df["OPEN_USER"] = df["OPEN_USER"].fillna("Unknown")
    
    # --- CLOSE_DATE: Drop rows without close date ---
    df = df.dropna(subset=["CLOSE_DATE"])
    print(f"   Dropped {initial_rows - len(df)} rows without CLOSE_DATE")
    
    # --- ESCALATED_GROUP ---
    df["ESCALATED_GROUP"] = df["ESCALATED_GROUP"].fillna("NO_GROUP")
    
    # --- CALLBACK_MECHANISM ---
    if "CALLBACK_MECHANISM" in df.columns:
        mode_val = df["CALLBACK_MECHANISM"].mode()[0]
        df["CALLBACK_MECHANISM"] = df["CALLBACK_MECHANISM"].fillna(mode_val)
        print(f"   Filled CALLBACK_MECHANISM with mode: '{mode_val}'")
    
    # --- OFFER_NAME & CUSTOMER_GROUP: Context-based imputation ---
    # TCRMService users typically don't have offer names
    tcrm_mask = df["OPEN_USER"] == "TCRMService"
    df.loc[tcrm_mask & df["OFFER_NAME"].isna(), "OFFER_NAME"] = "NO_OFFER_NAME"
    df.loc[tcrm_mask & df["CUSTOMER_GROUP"].isna(), "CUSTOMER_GROUP"] = "NO_GROUP"
    
    # FTTH Home offer -> FTTH Home group
    ftth_mask = df["OFFER_NAME"] == "FTTH Home"
    df.loc[ftth_mask & df["CUSTOMER_GROUP"].isna(), "CUSTOMER_GROUP"] = "FTTH Home"
    ftth_group_mask = df["CUSTOMER_GROUP"] == "FTTH Home"
    df.loc[ftth_group_mask & df["OFFER_NAME"].isna(), "OFFER_NAME"] = "FTTH Home"
    
    # Wanadoo -> Bitstream Home
    wanadoo_mask = df["CUSTOMER_GROUP"] == "Wanadoo-ADSL-Res"
    df.loc[wanadoo_mask & df["OFFER_NAME"].isna(), "OFFER_NAME"] = "Bitstream Home"
    
    # Orange Gov 10 CC -> PM Governorate
    gov_mask = df["OFFER_NAME"] == "Orange Gov 10 CC"
    df.loc[gov_mask & df["CUSTOMER_GROUP"].isna(), "CUSTOMER_GROUP"] = "PM Governorate"
    
    # Fill remaining CUSTOMER_GROUP with 'Prepaid'
    df["CUSTOMER_GROUP"] = df["CUSTOMER_GROUP"].fillna("Prepaid")
    
    # Drop remaining rows without OFFER_NAME
    rows_before = len(df)
    df = df.dropna(subset=["OFFER_NAME"])
    if rows_before - len(df) > 0:
        print(f"   Dropped {rows_before - len(df)} rows without OFFER_NAME")
    
    # --- OPEN_GR & CLOSE_GROUP: Cross-reference imputation ---
    df = _cross_fill_groups(df)
    
    # Fill remaining
    df["CLOSE_GROUP"] = df["CLOSE_GROUP"].fillna("NO_CLOSE_GROUP")
    df["OPEN_GR"] = df["OPEN_GR"].fillna("NO_OPEN_GR")
    
    # --- AGE_BRACKET ---
    if "AGE_BRACKET" in df.columns and df["AGE_BRACKET"].isnull().any():
        median_val = df["AGE_BRACKET"].median()
        df["AGE_BRACKET"] = df["AGE_BRACKET"].fillna(median_val)
        print(f"   Filled AGE_BRACKET with median: {median_val}")
    
    return df


def _cross_fill_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-reference fill OPEN_GR and CLOSE_GROUP using frequency analysis.
    
    For each unique OPEN_GR value, find the most frequent CLOSE_GROUP
    and fill missing CLOSE_GROUP values. Then do the reverse.
    """
    # Fill CLOSE_GROUP based on OPEN_GR
    for group in df["OPEN_GR"].dropna().unique():
        filtered = df[df["OPEN_GR"] == group]
        counts = filtered["CLOSE_GROUP"].value_counts()
        if not counts.empty:
            most_frequent = counts.index[0]
            mask = (df["OPEN_GR"] == group) & (df["CLOSE_GROUP"].isna())
            df.loc[mask, "CLOSE_GROUP"] = most_frequent
    
    # Fill OPEN_GR based on CLOSE_GROUP
    for group in df["CLOSE_GROUP"].dropna().unique():
        filtered = df[df["CLOSE_GROUP"] == group]
        counts = filtered["OPEN_GR"].value_counts()
        if not counts.empty:
            most_frequent = counts.index[0]
            mask = (df["CLOSE_GROUP"] == group) & (df["OPEN_GR"].isna())
            df.loc[mask, "OPEN_GR"] = most_frequent
    
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features from OPEN_DATE and CLOSE_DATE.
    
    New features:
    - RESOLUTION_TIME_HOURS: Time to resolve in hours
    - OPEN_HOUR: Hour of the day the complaint was opened
    - OPEN_DAY_OF_WEEK: Day of week (0=Monday, 6=Sunday)
    - OPEN_MONTH: Month the complaint was opened
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with additional time features.
    """
    print("\n🕐 Creating time-based features...")
    
    # Parse dates
    df["OPEN_DATE"] = pd.to_datetime(df["OPEN_DATE"], format="%d.%m.%Y %H:%M", errors="coerce")
    df["CLOSE_DATE"] = pd.to_datetime(df["CLOSE_DATE"], format="%d.%m.%Y %H:%M", errors="coerce")
    
    # Resolution time in hours
    df["RESOLUTION_TIME_HOURS"] = (
        (df["CLOSE_DATE"] - df["OPEN_DATE"]).dt.total_seconds() / 3600
    ).round(2)
    
    # Handle negative or zero resolution times
    df.loc[df["RESOLUTION_TIME_HOURS"] < 0, "RESOLUTION_TIME_HOURS"] = 0
    
    # Time features from OPEN_DATE
    df["OPEN_HOUR"] = df["OPEN_DATE"].dt.hour
    df["OPEN_DAY_OF_WEEK"] = df["OPEN_DATE"].dt.dayofweek
    df["OPEN_MONTH"] = df["OPEN_DATE"].dt.month
    
    # Drop original date columns (not needed for ML)
    df = df.drop(columns=["OPEN_DATE", "CLOSE_DATE"], errors="ignore")
    
    # Fill any NaN created during date parsing
    for col in ["RESOLUTION_TIME_HOURS", "OPEN_HOUR", "OPEN_DAY_OF_WEEK", "OPEN_MONTH"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    print("   Created: RESOLUTION_TIME_HOURS, OPEN_HOUR, OPEN_DAY_OF_WEEK, OPEN_MONTH")
    
    return df


def encode_categorical(df: pd.DataFrame, target_col: str = "COMPLAINT_TYPE"):
    """
    Encode categorical columns using LabelEncoder.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    target_col : str
        Name of the target column.
    
    Returns
    -------
    tuple
        (X, y, encoders_dict) - Features, target, and fitted encoders.
    """
    from sklearn.preprocessing import LabelEncoder
    
    print("\n🏷️  Encoding categorical features...")
    
    # Drop CASE_ID - it's just an identifier
    if "CASE_ID" in df.columns:
        df = df.drop(columns=["CASE_ID"])
    
    encoders = {}
    cat_columns = df.select_dtypes(include=["object"]).columns
    
    for col in cat_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    
    print(f"   Encoded {len(cat_columns)} categorical columns")
    
    # Split features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    return X, y, encoders


def process_data(filepath: str):
    """
    Complete data processing pipeline.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    
    Returns
    -------
    tuple
        (X, y, encoders) - Processed features, target, and encoders.
    """
    print("=" * 60)
    print("🚀 COMPLAINT DATA PROCESSING PIPELINE")
    print("=" * 60)
    
    # Step 1: Load data
    df = load_data(filepath)
    
    # Step 2: Analyze missing values
    analyze_missing_values(df)
    
    # Step 3: Drop high-missing columns
    df = drop_high_missing_columns(df, threshold=85.0)
    
    # Step 4: Drop leaky columns (CASE, PRODUCT)
    df = drop_leaky_columns(df)
    
    # Step 5: Fill missing values
    df = fill_missing_values(df)
    
    # Step 6: Verify no missing values remain
    remaining_missing = df.isnull().sum().sum()
    if remaining_missing > 0:
        print(f"\n⚠️  Still {remaining_missing} missing values. Dropping remaining NaN rows...")
        df = df.dropna()
    
    # Step 7: Add time features
    df = add_time_features(df)
    
    # Step 8: Encode categorical features
    X, y, encoders = encode_categorical(df)
    
    print(f"\n✅ Processing complete!")
    print(f"   Features shape: {X.shape}")
    print(f"   Target distribution: {dict(pd.Series(y).value_counts())}")
    print("=" * 60)
    
    return X, y, encoders
