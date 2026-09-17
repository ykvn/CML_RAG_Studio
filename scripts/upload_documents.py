"""
OTCS to RAG Document Sync Tool
==============================

Prerequisites:
    pip install "httpx[http2]"

Environment Variables (Recommended):
    Linux / macOS:
        export OTCS_USERNAME="your_username"
        export OTCS_PASSWORD="your_password"
        export RAG_API_KEY="your_rag_bearer_token"

    Windows PowerShell:
        $env:OTCS_USERNAME="your_username"
        $env:OTCS_PASSWORD="your_password"
        $env:RAG_API_KEY="your_rag_bearer_token"

    Windows Command Prompt (CMD):
        set OTCS_USERNAME=your_username
        set OTCS_PASSWORD=your_password
        set RAG_API_KEY=your_rag_bearer_token

Usage Syntax:
    python upload_documents.py <data_source_id> --file <path_to_id_file> [OPTIONAL_PARAMS]

Parameters:
    Positional Arguments (Required):
        data_source_id        Target RAG DataSource ID (e.g. 961).

    Named Arguments:
        -f, --file            [REQUIRED] Path to text file containing DataIDs (one ID per line).
        --delay               [OPTIONAL] Delay in seconds between file downloads/uploads (default: 30).
        -h, --help            [OPTIONAL] Display help message and parameter list.

Execution Examples:
    1. Standard Run (Default 30s delay):
       python upload_documents.py 961 --file document_list.txt

    2. Custom Delay (10-second delay between processing files):
       python upload_documents.py 961 -f document_list.txt --delay 10
"""

import argparse
import getpass
import mimetypes
import os
import re
import shlex
import sys
import time
import urllib.parse
import httpx

if os.environ.get("JOB_ARGUMENTS"):
    sys.argv = [sys.argv[0]] + shlex.split(os.environ["JOB_ARGUMENTS"])

AUTH_URL = "http://contentecmdev.hq.bni.co.id/otcs/cs.exe/api/v1/auth"
DOWNLOAD_URL_TEMPLATE = "http://contentecmdev.hq.bni.co.id/otcs/cs.exe/api/v2/nodes/{node_id}/versions/{version_number}/content"
UPLOAD_URL_TEMPLATE = "https://rag.cai.apps.dataservices.bni.co.id/api/v1/rag/dataSources/{dataSourceId}/files"


def authenticate(client, username, password):
    payload = {
        "username": username,
        "password": password
    }
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }

    response = client.post(AUTH_URL, data=payload, headers=headers)
    response.raise_for_status()

    ticket = response.json().get("ticket")
    if not ticket:
        raise ValueError("Authentication ticket (otcsticket) not found in response.")

    return ticket


def get_filename_from_response(response, node_id):
    """Extracts filename from Content-Disposition header or defaults to node_id."""
    cd = response.headers.get("Content-Disposition")
    if cd:
        match = re.search(r'filename\*?=(?:"(.*?)"|([^;\s]+))', cd)
        if match:
            filename = match.group(1) or match.group(2)
            if filename.startswith("UTF-8''"):
                filename = urllib.parse.unquote(filename[7:])
            return filename
    return f"document_{node_id}.bin"


def load_node_ids(file_path):
    """Reads IDs line-by-line from a text file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def process_documents(client, data_source_id, ticket, bearer_token, node_ids, delay_seconds=30):
    upload_url = UPLOAD_URL_TEMPLATE.format(dataSourceId=data_source_id)
    headers_download = {
        "otcsticket": ticket,
        "User-Agent": "curl/8.7.1"
    }
    upload_headers = {
        "accept": "application/json",
        "User-Agent": "curl/8.7.1",
        "Authorization": f"Bearer {bearer_token}"
    }
    total_files = len(node_ids)

    for index, node_id in enumerate(node_ids, start=1):
        print(f"[-] [{index}/{total_files}] Processing DataID {node_id}...")

        # 1. Download file content from OTCS
        download_url = DOWNLOAD_URL_TEMPLATE.format(node_id=node_id, version_number=1)
        try:
            download_res = client.get(download_url, headers=headers_download)
            download_res.raise_for_status()

            # 2. Extract filename & dynamically detect content-type
            file_name = get_filename_from_response(download_res, node_id)
            guessed_type, _ = mimetypes.guess_type(file_name)
            content_type = guessed_type or download_res.headers.get("Content-Type") or "application/octet-stream"

            # 3. Stream directly to target RAG API
            files = {
                "file": (file_name, download_res.content, content_type)
            }

            upload_res = client.post(
                upload_url,
                headers=upload_headers,
                files=files,
                follow_redirects=False
            )

            # Detect unauthenticated SSO redirects
            if upload_res.status_code in (301, 302, 303, 307):
                print(f"  [X] Failed: Request redirected to {upload_res.headers.get('location')}. Verify Bearer token.")
                continue

            upload_res.raise_for_status()

            res_json = upload_res.json()
            doc_id = res_json[0].get("documentId", "N/A") if isinstance(res_json, list) and len(res_json) > 0 else "N/A"
            print(f"  [V] Successfully uploaded '{file_name}' ({content_type}) to RAG [Doc ID: {doc_id}]")

        except httpx.HTTPError as err:
            print(f"  [X] Error processing DataID {node_id}: {err}")

        # 4. Delay before processing next file
        if index < total_files:
            print(f"  [...] Waiting {delay_seconds} seconds before next request...")
            time.sleep(delay_seconds)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch documents from OTCS using an ID file and upload them to a target RAG Data Source.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("data_source_id", help="Target RAG DataSource ID")
    parser.add_argument("--file", "-f", required=True, help="Path to text file containing DataIDs")
    parser.add_argument("--delay", type=int, default=30, help="Delay in seconds between files (default: 30)")

    args = parser.parse_args()

    username = os.getenv("OTCS_USERNAME") or input("Enter OTCS Username: ").strip()
    password = os.getenv("OTCS_PASSWORD") or getpass.getpass("Enter OTCS Password: ")
    bearer_token = os.getenv("RAG_API_KEY") or getpass.getpass("Enter RAG Bearer Token: ")

    try:
        node_ids = load_node_ids(args.file)
        print(f"[*] Loaded {len(node_ids)} document ID(s) from '{args.file}'.")

        with httpx.Client(http2=True, verify=False, timeout=60.0) as client:
            print("[*] Authenticating with OTCS...")
            ticket = authenticate(client, username, password)
            print("[*] Authentication successful.")

            print(f"[*] Uploading files to dataSourceId: {args.data_source_id}")
            process_documents(client, args.data_source_id, ticket, bearer_token, node_ids, delay_seconds=args.delay)
            print("[*] Process completed successfully.")

    except Exception as err:
        print(f"[!] Fatal error: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()