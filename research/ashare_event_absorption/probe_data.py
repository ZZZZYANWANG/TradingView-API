from __future__ import annotations

import json
import traceback
from pathlib import Path

OUT = Path("research_outputs/ashare_event_absorption_probe")
OUT.mkdir(parents=True, exist_ok=True)


def frame_info(frame):
    return {
        "shape": list(frame.shape),
        "columns": [str(c) for c in frame.columns],
        "head": frame.head(3).astype(str).to_dict(orient="records"),
    }


def main():
    report = {}
    try:
        import akshare as ak
        report["akshare_version"] = getattr(ak, "__version__", "unknown")
        for date in ["20231231", "20240630", "20241231", "20250331"]:
            key = f"stock_yjbb_em_{date}"
            try:
                df = ak.stock_yjbb_em(date=date)
                report[key] = frame_info(df)
                if len(df):
                    df.head(100).to_csv(OUT / f"{key}.csv", index=False)
                    break
            except Exception as exc:
                report[key] = {"error": repr(exc), "traceback": traceback.format_exc()[-3000:]}
        for func_name, kwargs in [
            ("stock_yjkb_em", {"date": "20241231"}),
            ("stock_yjyg_em", {"date": "20241231"}),
            ("index_stock_cons", {"symbol": "000300"}),
        ]:
            try:
                func = getattr(ak, func_name)
                df = func(**kwargs)
                report[func_name] = frame_info(df)
            except Exception as exc:
                report[func_name] = {"error": repr(exc), "traceback": traceback.format_exc()[-3000:]}
    except Exception as exc:
        report["akshare_import"] = {"error": repr(exc), "traceback": traceback.format_exc()[-3000:]}

    try:
        import baostock as bs
        report["baostock_login"] = str(bs.login())
        rs = bs.query_history_k_data_plus(
            "sh.600000",
            "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg,tradestatus,isST",
            start_date="2024-01-01",
            end_date="2024-03-31",
            frequency="d",
            adjustflag="2",
        )
        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())
        report["baostock_daily"] = {
            "error_code": rs.error_code,
            "error_msg": rs.error_msg,
            "fields": rs.fields,
            "rows": len(rows),
            "head": rows[:3],
        }
        rs60 = bs.query_history_k_data_plus(
            "sh.600000",
            "date,time,code,open,high,low,close,volume,amount,adjustflag",
            start_date="2024-01-02",
            end_date="2024-01-10",
            frequency="60",
            adjustflag="2",
        )
        rows60 = []
        while rs60.error_code == "0" and rs60.next():
            rows60.append(rs60.get_row_data())
        report["baostock_60m"] = {
            "error_code": rs60.error_code,
            "error_msg": rs60.error_msg,
            "fields": rs60.fields,
            "rows": len(rows60),
            "head": rows60[:5],
        }
        bs.logout()
    except Exception as exc:
        report["baostock"] = {"error": repr(exc), "traceback": traceback.format_exc()[-3000:]}

    (OUT / "probe.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
