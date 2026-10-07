"""Tests for spark_feature_push_client.utils.helpers (renamed offline columns)."""

import importlib
import sys

import pytest

from spark_feature_push_client.utils import helpers as helpers_module


def _metadata(storage_provider, source_base_path, fg_label="fg1", feature="age"):
    return {
        "keys": {"user": ["user_id"]},
        "data": [
            {
                "storage-provider": storage_provider,
                "base-path": [
                    {
                        "source-base-path": source_base_path,
                        "data-paths": [
                            {
                                "feature-group-label": fg_label,
                                "data-type": "INT32",
                                "source-data-column": "age_col",
                                "feature-label": feature,
                                "default-value": "0",
                            }
                        ],
                    }
                ],
            }
        ],
    }


@pytest.fixture(params=["shared", "fallback"])
def helpers(request, monkeypatch):
    """Run every test against both the bharatml_commons path and the local fallback."""
    if request.param == "shared":
        yield helpers_module
        return

    # Make `import bharatml_commons...` fail so that the module takes its ImportError branch.
    for name in list(sys.modules):
        if name == "bharatml_commons" or name.startswith("bharatml_commons."):
            monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setitem(sys.modules, "bharatml_commons", None)
    reloaded = importlib.reload(helpers_module)
    assert reloaded._USE_SHARED_UTILS is False
    yield reloaded
    monkeypatch.undo()
    importlib.reload(helpers_module)


@pytest.mark.parametrize(
    "storage_provider, path",
    [
        ("PARQUET_GCS", "gs://bucket/dir/User-Features/"),
        ("PARQUET_S3", "s3://bucket/dir/User-Features/"),
        ("DELTA_S3", "s3a://bucket/dir/User-Features"),
        ("DELTA_ADLS", "abfss://c@acct.dfs.core.windows.net/dir/User-Features"),
        ("PARQUET_ADLS", "abfss://c@acct.dfs.core.windows.net/dir/User-Features"),
        ("DELTA_GCS", "gs://bucket/dir/User-Features"),
    ],
)
def test_path_sources_are_renamed_for_every_cloud(helpers, storage_provider, path):
    feature_list, defaults, onfs_map, ofs_map, *_ = (
        helpers.get_features_from_response_v2(_metadata(storage_provider, path), [])
    )

    assert ofs_map == {"fg1": ["user_features___fg1___age"]}
    assert onfs_map == {"fg1": ["age"]}
    assert defaults == {"user_features___fg1___age": "0"}
    assert feature_list == [
        [path, storage_provider, [("age_col", "user_features___fg1___age")]]
    ]


@pytest.mark.parametrize(
    "table_name, expected_prefix",
    [("db.events", "events"), ("events", "events"), ("cat.db.events", "events")],
)
def test_table_sources(helpers, table_name, expected_prefix):
    _, _, _, ofs_map, *_ = helpers.get_features_from_response_v2(
        _metadata("TABLE", table_name), []
    )

    assert ofs_map == {"fg1": [expected_prefix + "___fg1___age"]}


@pytest.mark.parametrize(
    "storage_provider, path",
    [("PARQUET_S3", "s3://"), ("TABLE", "db."), ("CSV_LOCAL", "/tmp/x.csv")],
)
def test_unusable_source_exits_with_message(helpers, capsys, storage_provider, path):
    with pytest.raises(SystemExit) as exc:
        helpers.get_features_from_response_v2(_metadata(storage_provider, path), [])

    assert exc.value.code == 1
    assert "not expected" in capsys.readouterr().out


def test_fallback_clean_column_name_handles_empty_name(helpers):
    assert helpers.clean_column_name("") == ""
