from pathlib import Path

from dotenv import find_dotenv, load_dotenv


def ensure_env_loaded() -> None:
    """Load the project .env file once, regardless of the current working directory."""
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path=dotenv_path, override=False)
        return

    project_root = Path(__file__).resolve().parent.parent
    fallback = project_root / ".env"
    if fallback.exists():
        load_dotenv(dotenv_path=fallback, override=False)
