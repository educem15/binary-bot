"""
Cloud Storage
-------------
Uploads finished MP4 files to a publicly accessible URL so Instagram
can ingest them (Instagram requires a public URL, not a local file).

Supported backends (configure ONE in .env):
  1. AWS S3 / Cloudflare R2  – recommended for production
  2. Backblaze B2             – cheap ($0.006/GB stored, free egress)
  3. transfer.sh              – free, no signup, files live 14 days (dev/testing only)

The backend is selected automatically based on which env vars are set.
Priority: S3 > Backblaze > transfer.sh
"""

import logging
import os
from typing import Optional

import requests

log = logging.getLogger(__name__)


class CloudStorage:
    """Upload a local file and return its public URL."""

    def __init__(self):
        self.backend = self._detect_backend()
        log.info("Cloud storage backend: %s", self.backend)

    def upload(self, file_path: str) -> str:
        """Upload file and return public URL. Raises on failure."""
        if self.backend == "s3":
            return self._upload_s3(file_path)
        elif self.backend == "b2":
            return self._upload_b2(file_path)
        else:
            return self._upload_transfer_sh(file_path)

    # ── Backend detection ──────────────────────────────────────────────────

    def _detect_backend(self) -> str:
        if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_S3_BUCKET"):
            return "s3"
        if os.getenv("B2_KEY_ID") and os.getenv("B2_APPLICATION_KEY") and os.getenv("B2_BUCKET_NAME"):
            return "b2"
        return "transfer_sh"

    # ── S3 / Cloudflare R2 ─────────────────────────────────────────────────
    # Works for both AWS S3 and Cloudflare R2 (R2 is S3-compatible, much cheaper)
    # Cloudflare R2: $0.015/GB stored, zero egress cost
    # AWS S3:        $0.023/GB stored + egress
    #
    # Setup:
    #  AWS S3:        https://aws.amazon.com/s3/
    #  Cloudflare R2: https://developers.cloudflare.com/r2/
    #
    # Required env vars:
    #   AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_BUCKET
    #   AWS_S3_REGION (default: us-east-1)
    #   AWS_S3_ENDPOINT_URL  (only for R2: https://<account_id>.r2.cloudflarestorage.com)
    #   AWS_S3_PUBLIC_URL    (your bucket's public base URL)

    def _upload_s3(self, file_path: str) -> str:
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError

        bucket = os.environ["AWS_S3_BUCKET"]
        region = os.getenv("AWS_S3_REGION", "us-east-1")
        endpoint = os.getenv("AWS_S3_ENDPOINT_URL")      # None for AWS, set for R2
        public_base = os.getenv("AWS_S3_PUBLIC_URL", f"https://{bucket}.s3.{region}.amazonaws.com")

        s3 = boto3.client(
            "s3",
            region_name=region,
            endpoint_url=endpoint,
        )
        key = f"ugc-videos/{os.path.basename(file_path)}"
        s3.upload_file(
            file_path, bucket, key,
            ExtraArgs={"ContentType": "video/mp4", "ACL": "public-read"},
        )
        url = f"{public_base.rstrip('/')}/{key}"
        log.info("Uploaded to S3: %s", url)
        return url

    # ── Backblaze B2 ──────────────────────────────────────────────────────
    # Very cheap: $0.006/GB stored, $0.01/GB egress (or free via Cloudflare CDN)
    # Setup: https://www.backblaze.com/b2/sign-up.html (10 GB free tier)
    # Required env vars: B2_KEY_ID, B2_APPLICATION_KEY, B2_BUCKET_NAME, B2_BUCKET_URL

    def _upload_b2(self, file_path: str) -> str:
        key_id = os.environ["B2_KEY_ID"]
        app_key = os.environ["B2_APPLICATION_KEY"]
        bucket_name = os.environ["B2_BUCKET_NAME"]
        bucket_url = os.environ["B2_BUCKET_URL"]  # e.g. https://f005.backblazeb2.com

        # Step 1 – Authorise
        auth = requests.get(
            "https://api.backblazeb2.com/b2api/v2/b2_authorize_account",
            auth=(key_id, app_key),
            timeout=10,
        )
        auth.raise_for_status()
        auth_data = auth.json()
        api_url = auth_data["apiUrl"]
        auth_token = auth_data["authorizationToken"]
        download_url = auth_data["downloadUrl"]

        # Step 2 – Get upload URL
        upload_url_resp = requests.post(
            f"{api_url}/b2api/v2/b2_get_upload_url",
            headers={"Authorization": auth_token},
            json={"bucketId": self._get_b2_bucket_id(api_url, auth_token, bucket_name)},
            timeout=10,
        )
        upload_url_resp.raise_for_status()
        upload_data = upload_url_resp.json()

        # Step 3 – Upload
        file_name = f"ugc-videos/{os.path.basename(file_path)}"
        with open(file_path, "rb") as f:
            file_bytes = f.read()

        up = requests.post(
            upload_data["uploadUrl"],
            headers={
                "Authorization": upload_data["authorizationToken"],
                "X-Bz-File-Name": requests.utils.quote(file_name),
                "Content-Type": "video/mp4",
                "Content-Length": str(len(file_bytes)),
                "X-Bz-Content-Sha1": "do_not_verify",
            },
            data=file_bytes,
            timeout=120,
        )
        up.raise_for_status()

        url = f"{download_url}/file/{bucket_name}/{file_name}"
        log.info("Uploaded to B2: %s", url)
        return url

    def _get_b2_bucket_id(self, api_url: str, auth_token: str, bucket_name: str) -> str:
        resp = requests.post(
            f"{api_url}/b2api/v2/b2_list_buckets",
            headers={"Authorization": auth_token},
            json={"accountId": "", "bucketName": bucket_name, "bucketTypes": ["allPublic"]},
            timeout=10,
        )
        resp.raise_for_status()
        buckets = resp.json().get("buckets", [])
        if not buckets:
            raise ValueError(f"B2 bucket '{bucket_name}' not found")
        return buckets[0]["bucketId"]

    # ── transfer.sh (free, no signup, dev/test only) ───────────────────────
    # Files survive 14 days, max 10 GB, max 10 downloads.
    # DO NOT use for production – links expire.

    def _upload_transfer_sh(self, file_path: str) -> str:
        log.warning(
            "Using transfer.sh (DEV ONLY) – links expire in 14 days. "
            "Set up S3 or B2 for production."
        )
        file_name = os.path.basename(file_path)
        with open(file_path, "rb") as f:
            resp = requests.put(
                f"https://transfer.sh/{file_name}",
                data=f,
                headers={"Max-Downloads": "10", "Max-Days": "14"},
                timeout=120,
            )
        resp.raise_for_status()
        url = resp.text.strip()
        log.info("Uploaded to transfer.sh: %s", url)
        return url
