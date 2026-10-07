"""Tests for bharatml_commons.column_utils."""

import pytest

from bharatml_commons.column_utils import clean_column_name, generate_renamed_column

PATH_SOURCE_TYPES = [
    "PARQUET_GCS",
    "PARQUET_S3",
    "PARQUET_ADLS",
    "DELTA_GCS",
    "DELTA_S3",
    "DELTA_ADLS",
]


class TestCleanColumnName:
    def test_replaces_invalid_characters_and_lowercases(self):
        assert clean_column_name("Feature@Name#1") == "feature_name_1"

    def test_prefixes_leading_digit(self):
        assert clean_column_name("1abc") == "_1abc"

    def test_valid_name_is_unchanged(self):
        assert clean_column_name("already_valid_9") == "already_valid_9"

    def test_empty_name_does_not_raise(self):
        assert clean_column_name("") == ""


class TestGenerateRenamedColumnForTables:
    def test_schema_qualified_table(self):
        assert (
            generate_renamed_column("db.tbl", "TABLE", "fg", "feat")
            == "tbl___fg___feat"
        )

    def test_unqualified_table(self):
        assert (
            generate_renamed_column("tbl", "TABLE", "fg", "feat") == "tbl___fg___feat"
        )

    def test_table_name_is_not_cleaned(self):
        # TABLE names have always been used verbatim, only paths are normalised.
        assert (
            generate_renamed_column("db.My_Tbl", "TABLE", "fg", "feat")
            == "My_Tbl___fg___feat"
        )


class TestGenerateRenamedColumnForPaths:
    @pytest.mark.parametrize("source_type", PATH_SOURCE_TYPES)
    @pytest.mark.parametrize(
        "path",
        [
            "gs://bucket/warehouse/My-Table",
            "s3://bucket/warehouse/My-Table",
            "s3a://bucket/warehouse/My-Table",
            "abfss://container@account.dfs.core.windows.net/warehouse/My-Table",
        ],
    )
    def test_last_path_segment_is_used_for_every_scheme(self, source_type, path):
        assert (
            generate_renamed_column(path, source_type, "fg", "feat")
            == "my_table___fg___feat"
        )

    def test_trailing_slash_is_ignored(self):
        assert (
            generate_renamed_column("s3://bucket/a/b/", "DELTA_S3", "fg", "feat")
            == "b___fg___feat"
        )

    def test_gcs_format_is_unchanged(self):
        # Format produced before the scheme handling was generalised.
        assert (
            generate_renamed_column(
                "gs://bkt/path/to/user_features", "PARQUET_GCS", "fg1", "age"
            )
            == "user_features___fg1___age"
        )

    def test_bucket_root_uses_bucket_name(self):
        assert (
            generate_renamed_column("gs://bucket", "DELTA_GCS", "fg", "feat")
            == "bucket___fg___feat"
        )


class TestGenerateRenamedColumnErrors:
    def test_unsupported_source_type(self):
        with pytest.raises(ValueError, match="Unsupported source type"):
            generate_renamed_column("db.tbl", "CSV_LOCAL", "fg", "feat")

    @pytest.mark.parametrize("table_name", ["", "   ", "db.", ".tbl", "db..tbl"])
    def test_malformed_table_name(self, table_name):
        with pytest.raises(ValueError):
            generate_renamed_column(table_name, "TABLE", "fg", "feat")

    @pytest.mark.parametrize("path", ["", "   ", "gs://", "s3://", "s3:///"])
    def test_malformed_path(self, path):
        with pytest.raises(ValueError):
            generate_renamed_column(path, "PARQUET_S3", "fg", "feat")
