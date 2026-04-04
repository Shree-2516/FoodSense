from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

ROOT_APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
SPEC = spec_from_file_location("root_entry_app", ROOT_APP_PATH)

if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Unable to load root app from {ROOT_APP_PATH}")

root_app = module_from_spec(SPEC)
SPEC.loader.exec_module(root_app)

app = root_app.app

if __name__ == "__main__":
    app.run(debug=True)
