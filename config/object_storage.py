"""
Object storage utilities for TradingAgents.

Supports both S3 (production) and local filesystem (development) based on
the OBJECT_STORAGE_TYPE environment variable.

Environment Variables:
    OBJECT_STORAGE_TYPE: 's3' or 'fs' (default: 'fs')
    BUCKET_NAME: S3 bucket name (required if using S3)
    STORAGE_ROOT: Local storage directory (default: './data/storage')
    AWS_REGION: AWS region (default: 'us-east-1')
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv()

# Module-level variables for configuration
_storage_type: Optional[str] = None
_s3_client = None
_bucket_name: Optional[str] = None
_storage_root: Optional[Path] = None


def _init_storage():
    """Initialize storage configuration on first use"""
    global _storage_type, _s3_client, _bucket_name, _storage_root

    if _storage_type is not None:
        return

    _storage_type = os.getenv("OBJECT_STORAGE_TYPE_DEV", os.getenv("OBJECT_STORAGE_TYPE", "fs")).lower()

    if _storage_type == "s3":
        import boto3
        _s3_client = boto3.client('s3', region_name=os.getenv("AWS_REGION", "us-east-1"))
        _bucket_name = os.getenv("BUCKET_NAME")
        if not _bucket_name:
            raise ValueError("BUCKET_NAME environment variable required for S3 storage")
        logger.info(f"Initialized S3 object storage: {_bucket_name}")
    else:
        _storage_root = Path(os.getenv("STORAGE_ROOT", "./data/storage"))
        _storage_root.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized filesystem object storage: {_storage_root}")


def get_object(
    key: str,
    decode: bool = True,
    parse_json: bool = False
) -> Optional[Union[str, bytes, Dict[str, Any]]]:
    """
    Get object content

    Args:
        key: Object key/path
        decode: If True, decode bytes to string (default: True)
        parse_json: If True, parse content as JSON (default: False)

    Returns:
        Object content as string, bytes, or dict (if parse_json=True)
        Returns None if object doesn't exist or error occurs

    Examples:
        # Get text content
        content = get_object('data/file.txt')

        # Get JSON content
        data = get_object('data/config.json', parse_json=True)

        # Get raw bytes
        raw_data = get_object('data/image.png', decode=False)
    """
    _init_storage()

    try:
        if _storage_type == "s3":
            response = _s3_client.get_object(Bucket=_bucket_name, Key=key)
            content = response['Body'].read()
            logger.info(f"Retrieved object from s3://{_bucket_name}/{key}")
        else:
            file_path = _storage_root / key
            if not file_path.exists():
                logger.warning(f"Object not found: {file_path}")
                return None
            content = file_path.read_bytes()
            logger.info(f"Retrieved object from {file_path}")

        if decode:
            content = content.decode('utf-8')
            if parse_json:
                content = json.loads(content)

        return content

    except Exception as e:
        logger.error(f"Error getting object '{key}': {e}")
        return None


def put_object(
    key: str,
    content: Union[str, bytes, Dict[str, Any]],
    content_type: Optional[str] = None
) -> bool:
    """
    Put object content

    Args:
        key: Object key/path
        content: Content to store (string, bytes, or dict for JSON)
        content_type: MIME type of content (optional)

    Returns:
        True if successful, False otherwise

    Examples:
        # Store text
        put_object('data/file.txt', 'Hello World')

        # Store JSON
        put_object('data/config.json', {'key': 'value'})

        # Store bytes
        put_object('data/image.png', image_bytes, content_type='image/png')
    """
    _init_storage()

    try:
        # Convert dict to JSON string
        if isinstance(content, dict):
            content = json.dumps(content, indent=2)
            if content_type is None:
                content_type = 'application/json'

        # Convert string to bytes
        if isinstance(content, str):
            body = content.encode('utf-8')
            if content_type is None:
                content_type = 'text/plain'
        else:
            body = content
            if content_type is None:
                content_type = 'application/octet-stream'

        if _storage_type == "s3":
            kwargs = {
                'Bucket': _bucket_name,
                'Key': key,
                'Body': body,
            }
            if content_type:
                kwargs['ContentType'] = content_type

            _s3_client.put_object(**kwargs)
            logger.info(f"Uploaded object to s3://{_bucket_name}/{key}")
        else:
            file_path = _storage_root / key
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(body)
            logger.info(f"Wrote object to {file_path}")

        return True

    except Exception as e:
        logger.error(f"Error putting object '{key}': {e}")
        return False


def list_objects(
    prefix: str = "",
    max_keys: int = 1000
) -> List[str]:
    """
    List objects with optional prefix

    Args:
        prefix: Key prefix to filter objects (default: "")
        max_keys: Maximum number of keys to return (default: 1000)

    Returns:
        List of object keys

    Examples:
        # List all objects
        all_files = list_objects()

        # List objects with prefix
        reports = list_objects(prefix='AAPL/2025-01-15/')
    """
    _init_storage()
    objects = []

    try:
        if _storage_type == "s3":
            paginator = _s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(
                Bucket=_bucket_name,
                Prefix=prefix,
                PaginationConfig={'MaxItems': max_keys}
            )

            for page in pages:
                if 'Contents' in page:
                    objects.extend([obj['Key'] for obj in page['Contents']])

            logger.info(f"Listed {len(objects)} objects from s3://{_bucket_name}/{prefix}")
        else:
            search_path = _storage_root / prefix if prefix else _storage_root
            if search_path.exists():
                for file_path in search_path.rglob('*'):
                    if file_path.is_file():
                        # Get relative path from storage root
                        rel_path = file_path.relative_to(_storage_root)
                        objects.append(str(rel_path).replace('\\', '/'))

                        if len(objects) >= max_keys:
                            break

            logger.info(f"Listed {len(objects)} objects from {search_path}")

        return objects

    except Exception as e:
        logger.error(f"Error listing objects with prefix '{prefix}': {e}")
        return []


def upload_directory(
    local_dir: Path,
    prefix: str
) -> List[str]:
    """
    Upload an entire directory, preserving directory structure

    Args:
        local_dir: Local directory path to upload
        prefix: Key prefix (e.g., "SPY/2025-01-15")

    Returns:
        List of uploaded keys

    Examples:
        # Upload analysis results
        uploaded = upload_directory(
            Path('/tmp/analysis'),
            'AAPL/2025-01-15'
        )
    """
    _init_storage()
    uploaded_files = []

    local_dir = Path(local_dir)

    if not local_dir.exists():
        logger.warning(f"Local directory does not exist: {local_dir}")
        return uploaded_files

    try:
        # Walk through all files in the directory
        for file_path in local_dir.rglob('*'):
            if file_path.is_file():
                # Calculate relative path from local_dir
                relative_path = file_path.relative_to(local_dir)

                # Construct key: {prefix}/{relative_path}
                key = f"{prefix}/{relative_path}".replace('\\', '/')

                # Read file content
                content = file_path.read_bytes()

                # Upload using put_object
                if put_object(key, content):
                    uploaded_files.append(key)

        logger.info(f"Uploaded {len(uploaded_files)} files from {local_dir}")
        return uploaded_files

    except Exception as e:
        logger.error(f"Error uploading directory '{local_dir}': {e}")
        return uploaded_files
