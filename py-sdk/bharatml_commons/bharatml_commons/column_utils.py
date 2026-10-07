"""
Column and data processing utilities for BharatML Stack SDKs
"""

import re
from typing import List, Tuple


def clean_column_name(column_name: str) -> str:
    """
    Cleans a column name to be Delta Table compatible:
    - Replaces unsupported special characters with "_"
    - Ensures it doesn't start with a number
    - Converts to lowercase

    Args:
        column_name: Original column name to clean

    Returns:
        Cleaned column name compatible with Delta Table standards
    """
    # Define a regex pattern to match invalid characters
    invalid_chars = r"[^a-zA-Z0-9_]"
    cleaned_name = re.sub(invalid_chars, "_", column_name)  # Replace invalid chars with "_"

    # Ensure it doesn't start with a number
    if cleaned_name and cleaned_name[0].isdigit():
        cleaned_name = "_" + cleaned_name

    return cleaned_name.lower()  # Convert to lowercase (Delta standard)


# Storage URI scheme, e.g. "gs://", "s3://", "s3a://", "abfss://".
_URI_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*://")

_PATH_SOURCE_TYPES = (
    "PARQUET_GCS", "PARQUET_S3", "PARQUET_ADLS",  # Parquet sources
    "DELTA_GCS", "DELTA_S3", "DELTA_ADLS",        # Delta sources
)


def _last_path_segment(path: str) -> str:
    """Return the last non-empty segment of a storage path, ignoring the URI scheme."""
    without_scheme = _URI_SCHEME_RE.sub("", path.strip(), count=1)
    return without_scheme.strip("/ ").split("/")[-1]


def generate_renamed_column(table_name: str, source_type: str, fg_label: str, feature_col: str) -> str:
    """
    Generate renamed column based on source type and table name

    Args:
        table_name: Name of the source table/path. For TABLE sources a (possibly
            schema-qualified) table name, for PARQUET_*/DELTA_* sources a storage
            path such as gs://bucket/dir, s3://bucket/dir or
            abfss://container@account.dfs.core.windows.net/dir
        source_type: Type of data source (TABLE, PARQUET_*, DELTA_*)
        fg_label: Feature group label
        feature_col: Original feature column name

    Returns:
        Renamed column following naming conventions

    Raises:
        ValueError: If source_type is not supported or table_name is empty or malformed
    """
    if source_type == "TABLE":
        parts = table_name.strip().split(".")
        if not all(part.strip() for part in parts):
            raise ValueError(f"Invalid table name: {table_name!r}")
        table_part = parts[1] if len(parts) > 1 else parts[0]
        rename_feature_col = table_part + "___" + fg_label + "___" + feature_col
    elif source_type in _PATH_SOURCE_TYPES:
        # Use the last part of the path, whatever the storage scheme, and clean it
        clean_table_name = clean_column_name(_last_path_segment(table_name))
        if not clean_table_name:
            raise ValueError(f"Invalid source path: {table_name!r}")
        rename_feature_col = clean_table_name + "___" + fg_label + "___" + feature_col
    else:
        raise ValueError(f"Unsupported source type: {source_type} for table: {table_name}")
    return rename_feature_col


def test_function():
    print("test_function 23")
