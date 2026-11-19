from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
ROUND10 = HERE.parent / "ashare_fluid_round10"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(ROUND10) not in sys.path:
    sys.path.insert(0, str(ROUND10))

import run_characteristic_network as research  # noqa: E402

spec = importlib.util.spec_from_file_location("ashare_round10_yahoo", ROUND10 / "run_yahoo.py")
if spec is None or spec.loader is None:
    raise RuntimeError("cannot import round10 Yahoo source")
yahoo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(yahoo)


def load_available_receipt() -> list[str]:
    response = requests.get(yahoo.RECEIPT_URL, headers=yahoo.HEADERS, timeout=30)
    response.raise_for_status()
    payload = response.json()
    codes = sorted({str(item["code"]) for item in payload if item.get("ok")})
    if len(codes) < 60:
        raise RuntimeError(f"historical-membership receipt too small even for replication: {len(codes)}")
    return codes


yahoo.load_code_receipt = load_available_receipt
research.source.load_baostock = yahoo.load_yahoo
research.OUT = Path("research_outputs/ashare_fluid_round15_yahoo")
research.OUT.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    research.main()
