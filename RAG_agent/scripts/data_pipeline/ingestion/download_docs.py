"""Script to download LangGraph and LangChain documentation files."""

import sys
from pathlib import Path

import httpx

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from settings import INPUT_DIR


def download_file(url: str, output_path: Path, timeout: float = 30.0) -> bool:
    """
    Download a file from a URL and save it to the specified path.

    Args:
        url: The URL to download from.
        output_path: The path where the file should be saved.
        timeout: Request timeout in seconds.

    Returns:
        True if download was successful, False otherwise.
    """
    try:
        print(f"Downloading {url}...")
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()

        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write file
        output_path.write_text(response.text, encoding="utf-8")
        print(f"[OK] Saved to {output_path}")
        return True

    except httpx.HTTPStatusError as e:
        print(f"✗ HTTP error {e.response.status_code} for {url}: {e}")
        return False
    except httpx.RequestError as e:
        print(f"✗ Request error for {url}: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error downloading {url}: {e}")
        return False


def main() -> int:
    """
    Download LangGraph and LangChain documentation files.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    urls = [
        ("https://langchain-ai.github.io/langgraph/llms-full.txt", "langgraph_llms_full.txt"),
        ("https://docs.langchain.com/llms-full.txt", "langchain_llms_full.txt"),
    ]

    success_count = 0
    for url, filename in urls:
        output_path = INPUT_DIR / filename
        if download_file(url, output_path):
            success_count += 1

    if success_count == len(urls):
        print(f"\n[OK] Successfully downloaded {success_count} file(s) to {INPUT_DIR}")
        return 0
    else:
        print(f"\n✗ Failed to download {len(urls) - success_count} file(s)")
        return 1



