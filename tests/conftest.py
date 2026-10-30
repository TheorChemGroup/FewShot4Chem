"""Pytest configuration: load optional settings (e.g. TABPFN_MODEL_CACHE_DIR) from a local .env."""

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv is optional for the tests
    pass
else:
    load_dotenv()
