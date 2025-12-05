"""Clone or update the httpx repository for indexing.

This script clones the httpx repository from GitHub if it doesn't exist,
or updates it if it already exists.

Usage:
    python scripts/data_pipeline/ingestion/clone_httpx_repo.py
"""

import logging
import subprocess
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from settings import HTTPX_REPO_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

HTTPX_REPO_URL = "https://github.com/encode/httpx.git"


def clone_or_update_repo() -> bool:
    """Clone or update the httpx repository.

    Returns:
        True if successful, False otherwise
    """
    try:
        if HTTPX_REPO_DIR.exists() and (HTTPX_REPO_DIR / ".git").exists():
            logger.info(f"Repository exists at {HTTPX_REPO_DIR}, updating...")

            # Pull latest changes
            result = subprocess.run(
                ["git", "-C", str(HTTPX_REPO_DIR), "pull"],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"Updated repository: {result.stdout.strip()}")

        else:
            logger.info(f"Cloning httpx repository to {HTTPX_REPO_DIR}...")

            # Ensure parent directory exists
            HTTPX_REPO_DIR.parent.mkdir(parents=True, exist_ok=True)

            # Clone repository
            result = subprocess.run(
                ["git", "clone", HTTPX_REPO_URL, str(HTTPX_REPO_DIR)],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info("Cloned repository successfully")

        # Get repository info
        result = subprocess.run(
            ["git", "-C", str(HTTPX_REPO_DIR), "log", "-1", "--format=%H %s"],
            capture_output=True,
            text=True,
            check=True
        )
        logger.info(f"Latest commit: {result.stdout.strip()}")

        # Count Python files
        python_files = list(HTTPX_REPO_DIR.rglob("*.py"))
        logger.info(f"Found {len(python_files)} Python files in repository")

        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Git command failed: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Failed to clone/update repository: {e}")
        return False


def main():
    """Main entry point."""
    logger.info("Starting httpx repository clone/update...")

    success = clone_or_update_repo()

    if success:
        logger.info("✅ Repository ready for indexing")
        sys.exit(0)
    else:
        logger.error("❌ Failed to prepare repository")
        sys.exit(1)


if __name__ == "__main__":
    main()
