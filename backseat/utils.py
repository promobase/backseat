import os


def load_dotenv(dotenv_path=".env"):
    """
    Loads environment variables from a .env file into os.environ.
    Each line in the .env file should be in KEY=VALUE format.
    Lines starting with # are treated as comments.
    """
    if not os.path.exists(dotenv_path):
        return

    with open(dotenv_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)
