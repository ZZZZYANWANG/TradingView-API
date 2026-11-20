from __future__ import annotations

import json
import traceback
from pathlib import Path

import requests

OUT = Path("research_outputs/ashare_event_absorption_probe")
OUT.mkdir(parents=True, exist_ok=True)


def eastmoney_probe(report_date: str) -> dict:
    url = "https://datacenter-web.eastmoney.com/api/data/v1/get"
    params = {
        "reportName": "RPT_LICO_FN_CPD",
        "columns": "ALL",
        "filter": f"(REPORTDATE='{report_date[:4]}-{report_date[4:6]}-{report_date[6:]}')",
        "pageNumber": 1,
        "pageSize": 10,
        "sortColumns": "NOTICE_DATE,SECURITY_CODE",
        "sortTypes": "-1,1",
        "source": "WEB",
        "client": "WEB",
    }
    response = requests.get(url, params=params, timeout=(10, 30))
    response.raise_for_status()
    payload = response.json()
    result = payload.get("result") or {}
    rows = result.get("data") or []
    return {
        "url": response.url,
        "success": payload.get("success"),
        "message": payload.get("message"),
        "pages": result.get("pages"),
        "count": result.get("count"),
        "keys": sorted(rows[0].keys()) if rows else [],
        "head": rows[:3],
    }


def main():
    report = {}
    for date in ["2024-12-31", "2024-06-30", "2023-12-31"]:
        key = f"eastmoney_{date}"
        try:
            report[key] = eastmoney_probe(date.replace("-", ""))
            if report[key].get("head"):
                break
        except Exception as exc:
            report[key] = {"error": repr(exc), "traceback": traceback.format_exc()[-3000:]}

    try:
        import baostock as bs
        login = bs.login()
        report["baostock_login"] = {"code": login.error_code, "msg": login.error_msg}
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
        for date in ["2019-12-31", "2024-12-31"]:
            for label, func in [("hs300", bs.query_hs300_stocks), ("zz500", bs.query_zz500_stocks)]:
                try:
                    rsu = func(date)
                    rowsu = []
                    while rsu.error_code == "0" and rsu.next():
                        rowsu.append(rsu.get_row_data())
                    report[f"{label}_{date}"] = {
                        "error_code": rsu.error_code,
                        "error_msg": rsu.error_msg,
                        "fields": rsu.fields,
                        "rows": len(rowsu),
                        "head": rowsu[:3],
                    }
                except Exception as exc:
                    report[f"{label}_{date}"] = {"error": repr(exc)}
        bs.logout()
    except Exception as exc:
        report["baostock"] = {"error": repr(exc), "traceback": traceback.format_exc()[-3000:]}

    (OUT / "probe.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
