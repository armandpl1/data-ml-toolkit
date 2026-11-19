"""Read and write objects in AWS S3.

Needs the `aws` extra: `data-ml-toolkit[aws]`. Credentials come from the
standard AWS chain (environment variables, ~/.aws, or an IAM role in CI and
on AWS), never from code.

    from data_ml_toolkit import s3
    s3.upload_file("data/processed/films.parquet", "s3://my-bucket/films.parquet")
    s3.download_file("s3://my-bucket/films.parquet", "data/raw/films.parquet")

Files are sent in one request (`put_object`), which suits the data and model
files in these repos; S3 caps a single request at 5 GB.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Split `s3://bucket/key/path` into ("bucket", "key/path")."""
    parsed = urlparse(uri)
    bucket, key = parsed.netloc, parsed.path.lstrip("/")
    if parsed.scheme != "s3" or not bucket:
        raise ValueError(f"Not an S3 URI (expected s3://bucket/key): {uri!r}")
    return bucket, key


def get_client(region: str | None = None) -> S3Client:
    """An S3 client from the standard AWS credential chain."""
    try:
        import boto3
    except ImportError as err:
        raise ImportError("data_ml_toolkit.s3 needs boto3: install data-ml-toolkit[aws]") from err
    return boto3.client("s3", region_name=region)


def upload_file(path: str | Path, uri: str, client: S3Client | None = None) -> None:
    """Upload a local file to an s3:// URI."""
    bucket, key = parse_s3_uri(uri)
    if not key:
        raise ValueError(f"S3 URI needs a key, not just a bucket: {uri!r}")
    client = client or get_client()
    client.put_object(Bucket=bucket, Key=key, Body=Path(path).read_bytes())


def download_file(uri: str, path: str | Path, client: S3Client | None = None) -> Path:
    """Download an s3:// object to a local file, creating parent folders."""
    bucket, key = parse_s3_uri(uri)
    client = client or get_client()
    body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return path


def list_keys(prefix_uri: str, client: S3Client | None = None) -> list[str]:
    """Every key under an s3:// prefix, following pagination."""
    bucket, prefix = parse_s3_uri(prefix_uri)
    client = client or get_client()
    keys: list[str] = []
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        keys.extend(obj["Key"] for obj in page.get("Contents", []))
    return keys
