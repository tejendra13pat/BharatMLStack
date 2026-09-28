"""
Regression tests for issue #165: GRPCFeatureClient sent gRPC metadata header
keys (``ONLINE-FEATURE-STORE-CALLER-ID`` / ``ONLINE-FEATURE-STORE-AUTH-TOKEN``)
containing uppercase letters, which grpcio's C-core rejects outright.

gRPC's HTTP/2 header-name grammar only allows lowercase ASCII letters,
digits, ``_``, ``.`` and ``-`` (see grpc-core's ``validate_metadata.cc``).
Hyphens are valid; the uppercase letters were the only problem. Sending a
call with an invalid key raises ``ValueError: metadata was invalid: ...``
from the grpcio bindings themselves, before any network I/O — a real
(no-mock) gRPC channel to an address nothing is listening on reproduces
this deterministically and quickly, without a live server.
"""

import re

import grpc
import pytest

from grpc_feature_client import GRPCClientConfig, GRPCFeatureClient
from grpc_feature_client.client import HEADER_CALLER_ID, HEADER_CALLER_TOKEN

# gRPC-core's own metadata key grammar (validate_metadata.cc): lowercase
# letters, digits, '_', '.', '-' only. Anything else -> "Illegal header key".
_VALID_GRPC_METADATA_KEY = re.compile(r"^[0-9a-z_.\-]+$")

# Nothing needs to be listening here: metadata-key validation happens
# client-side, before any connection attempt.
_UNREACHABLE_TARGET = "127.0.0.1:1"


@pytest.fixture
def client():
    config = GRPCClientConfig(
        server_address=_UNREACHABLE_TARGET,
        job_id="test-job-id",
        job_token="test-job-token",
    )
    return GRPCFeatureClient(config)


def test_auth_header_keys_match_grpc_metadata_grammar():
    """Whitebox: the header constants must themselves be legal gRPC metadata
    keys. This is gRPC-core's real validation grammar, not a guess."""
    assert _VALID_GRPC_METADATA_KEY.match(HEADER_CALLER_ID), HEADER_CALLER_ID
    assert _VALID_GRPC_METADATA_KEY.match(HEADER_CALLER_TOKEN), HEADER_CALLER_TOKEN


def test_created_metadata_is_accepted_by_grpc(client):
    """End-to-end regression test for the actual reported crash.

    Sends a real (unary_unary, no mocking) gRPC call using exactly the
    metadata ``_create_metadata()`` produces, against an address nothing is
    listening on. Before this fix, this raised
    ``ValueError: metadata was invalid: ...`` from the C-core bindings
    themselves -- the exact error from issue #165 -- because
    ``ONLINE-FEATURE-STORE-CALLER-ID`` / ``ONLINE-FEATURE-STORE-AUTH-TOKEN``
    contain uppercase letters. After the fix, the metadata is accepted and
    the call instead fails with a plain connection error (``UNAVAILABLE``),
    proving the header keys themselves are no longer the problem.
    """
    metadata = client._create_metadata()

    channel = grpc.insecure_channel(_UNREACHABLE_TARGET)
    call = channel.unary_unary("/bharatmlstack.test.Fake/Method")
    try:
        call(b"", metadata=metadata, timeout=2)
    except ValueError as e:
        pytest.fail(
            "gRPC rejected the metadata as malformed (this is the bug from "
            f"issue #165 reproducing): {e}"
        )
    except grpc.RpcError as e:
        # Expected: metadata was accepted; the call fails only because
        # nothing is listening on the target address.
        assert e.code() == grpc.StatusCode.UNAVAILABLE
    finally:
        channel.close()


def test_metadata_contains_expected_auth_headers(client):
    metadata = dict(client._create_metadata())
    assert metadata[HEADER_CALLER_ID] == "test-job-id"
    assert metadata[HEADER_CALLER_TOKEN] == "test-job-token"
