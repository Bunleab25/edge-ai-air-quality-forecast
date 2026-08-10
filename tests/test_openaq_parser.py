import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "00_collect_data.py"
SPEC = importlib.util.spec_from_file_location("collect_data_script", SCRIPT_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(module)


def test_result_date_supports_current_openaq_shape():
    payload = {
        "period": {
            "datetimeFrom": {"utc": "2021-04-29T17:00:00Z", "local": "2021-04-30T00:00:00+07:00"}
        }
    }

    assert module._result_date(payload) == "2021-04-30"
