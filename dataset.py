"""Utility module for loading and caching the Tagore songs dataset."""

from datasets import load_dataset
import pandas as pd
import os

_CACHE_PATH = os.path.join(os.path.dirname(__file__), '.tagore_cache.parquet')

# Column name mapping: Bengali → English
COLUMN_MAP = {
    'গানের কথা': 'lyrics',
    'পর্যায়': 'category',
    'ভাগ': 'subcategory',
    'ক্রমিক সংখ্যা': 'serial_number',
    'বাংলা_সন': 'bengali_year',
    'ইংরেজি_সন': 'gregorian_year',
    'রাগ': 'raga',
    'তাল': 'taal',
}

# English descriptions of each column for reference
COLUMN_DESCRIPTIONS = {
    'lyrics': 'Full Bengali song lyrics (Unicode text)',
    'category': 'Primary thematic classification (পর্যায়) — e.g. পূজা (Devotion), প্রেম (Love), প্রকৃতি (Nature), স্বদেশ (Patriotism), নাট্যগীতি (Dramatic)',
    'subcategory': 'Thematic subdivision (ভাগ) — 92 classes, finer grouping within category',
    'serial_number': 'Catalog/serial number in the Gitabitan collection',
    'bengali_year': 'Year of composition in Bengali calendar',
    'gregorian_year': 'Year of composition in Gregorian calendar',
    'raga': 'Melodic mode/framework (রাগ) — 394 distinct ragas from Hindustani classical music',
    'taal': 'Rhythmic cycle pattern (তাল) — 79 distinct patterns',
}


def load_tagore() -> pd.DataFrame:
    """Load the Tagore songs dataset, rename columns to English, return a clean DataFrame.

    Returns a DataFrame with 2316 rows and columns:
        lyrics, category, subcategory, serial_number, bengali_year,
        gregorian_year, raga, taal
    """
    if os.path.exists(_CACHE_PATH):
        return pd.read_parquet(_CACHE_PATH)
    ds = load_dataset("Pradipta/tagore_songs", split="train")
    df = pd.DataFrame(ds)
    df = df.rename(columns=COLUMN_MAP)
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')
    df.to_parquet(_CACHE_PATH)
    return df


def describe_dataset(df: pd.DataFrame) -> str:
    """Return a text summary of the dataset for agent context."""
    lines = [
        f"Tagore Songs Dataset: {len(df)} songs",
        f"Columns: {', '.join(df.columns)}",
        "",
    ]
    for col in df.columns:
        n_missing = df[col].isna().sum()
        col_numeric = pd.to_numeric(df[col], errors='coerce')
        non_null_numeric = col_numeric.notna().sum()
        # Treat as numeric if most non-null values convert successfully
        if non_null_numeric > 0 and non_null_numeric >= (len(df) - n_missing) * 0.5:
            lines.append(f"  {col}: range [{col_numeric.min():.0f}, {col_numeric.max():.0f}], {n_missing} missing")
        else:
            n_unique = df[col].nunique()
            top3 = df[col].value_counts().head(3)
            top3_str = ', '.join(f'{v} ({c})' for v, c in top3.items())
            lines.append(f"  {col}: {n_unique} unique, {n_missing} missing. Top: {top3_str}")
    return '\n'.join(lines)


if __name__ == '__main__':
    df = load_tagore()
    print(describe_dataset(df))
