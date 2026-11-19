import io
from pathlib import Path

import boto3
import pytest
from botocore.response import StreamingBody
from botocore.stub import Stubber

from data_ml_toolkit import s3


@pytest.fixture
def client():  # type: ignore[no-untyped-def]
    # Fake credentials and region: the Stubber intercepts every call, so
    # nothing reaches AWS.
    return boto3.client(
        "s3",
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("s3://bucket/a/b.parquet", ("bucket", "a/b.parquet")),
        ("s3://bucket", ("bucket", "")),
        ("s3://bucket/", ("bucket", "")),
    ],
)
def test_parse_s3_uri(uri: str, expected: tuple[str, str]) -> None:
    assert s3.parse_s3_uri(uri) == expected


@pytest.mark.parametrize("uri", ["http://bucket/key", "bucket/key", "s3:///key", ""])
def test_parse_s3_uri_rejects_non_s3(uri: str) -> None:
    with pytest.raises(ValueError, match="Not an S3 URI"):
        s3.parse_s3_uri(uri)


def test_upload_file_puts_the_bytes(tmp_path: Path, client) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "f.txt"
    path.write_bytes(b"hello")
    with Stubber(client) as stub:
        stub.add_response("put_object", {}, {"Bucket": "b", "Key": "dir/f.txt", "Body": b"hello"})
        s3.upload_file(path, "s3://b/dir/f.txt", client=client)
        stub.assert_no_pending_responses()


def test_upload_needs_a_key(tmp_path: Path, client) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match="needs a key"):
        s3.upload_file(tmp_path / "f.txt", "s3://b", client=client)


def test_download_file_writes_and_creates_folders(tmp_path: Path, client) -> None:  # type: ignore[no-untyped-def]
    body = StreamingBody(io.BytesIO(b"data"), len(b"data"))
    with Stubber(client) as stub:
        stub.add_response("get_object", {"Body": body}, {"Bucket": "b", "Key": "k.bin"})
        out = s3.download_file("s3://b/k.bin", tmp_path / "new" / "k.bin", client=client)
    assert out.read_bytes() == b"data"


def test_list_keys_follows_pagination(client) -> None:  # type: ignore[no-untyped-def]
    with Stubber(client) as stub:
        stub.add_response(
            "list_objects_v2",
            {
                "Contents": [{"Key": "p/1"}],
                "IsTruncated": True,
                "NextContinuationToken": "t",
            },
            {"Bucket": "b", "Prefix": "p/"},
        )
        stub.add_response(
            "list_objects_v2",
            {"Contents": [{"Key": "p/2"}], "IsTruncated": False},
            {"Bucket": "b", "Prefix": "p/", "ContinuationToken": "t"},
        )
        assert s3.list_keys("s3://b/p/", client=client) == ["p/1", "p/2"]
