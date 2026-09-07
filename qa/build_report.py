"""Join the QA catalogue with the recorded run results into an Excel workbook.

Sheets:
  Summary      - pass rate, breakdown by layer / priority / severity, verdict
  Test Cases   - full catalogue with steps, expected, actual, status
  Defects      - failed or blocked cases only, ordered by priority
  Demo Checks  - the manual pre-demo checks automation cannot cover
"""
import json
import sys
from datetime import datetime
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

sys.path.insert(0, str(Path(__file__).parent))
from testcases import CASES  # noqa: E402

QA = Path(__file__).parent
RESULTS = QA / "results.json"
OUT = QA / "REFlect_AI_Test_Report.xlsx"

HDR_FILL = PatternFill("solid", fgColor="1F3864")
HDR_FONT = Font(color="FFFFFF", bold=True, size=11)
TITLE_FONT = Font(bold=True, size=15, color="1F3864")
STATUS_FILL = {
    "PASS": PatternFill("solid", fgColor="C6EFCE"),
    "FAIL": PatternFill("solid", fgColor="FFC7CE"),
    "BLOCKED": PatternFill("solid", fgColor="FFEB9C"),
    "NOT RUN": PatternFill("solid", fgColor="E7E6E6"),
}
STATUS_FONT = {
    "PASS": Font(color="006100", bold=True),
    "FAIL": Font(color="9C0006", bold=True),
    "BLOCKED": Font(color="9C6500", bold=True),
    "NOT RUN": Font(color="666666", bold=True),
}
PRIO_FONT = {"P0 - Blocker": Font(color="9C0006", bold=True), "P1 - High": Font(color="9C6500", bold=True)}
THIN = Border(*[Side(style="thin", color="BFBFBF")] * 4)


def _style_header(ws, row=1):
    for cell in ws[row]:
        if cell.value:
            cell.fill, cell.font = HDR_FILL, HDR_FONT
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    # A cell reference string, not ws.cell(): the latter materialises a blank row.
    ws.freeze_panes = f"A{row + 1}"


def _widths(ws, widths):
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w


def main():
    results = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {}
    rows = []
    for c in CASES:
        r = results.get(c["id"], {})
        rows.append({**c,
                     "status": r.get("status", "NOT RUN"),
                     "actual": r.get("actual") or ("Passed as expected." if r.get("status") == "PASS" else ""),
                     "detail": r.get("detail", ""),
                     "duration": r.get("duration_s", ""),
                     "test": r.get("test", "")})

    total = len(rows)
    counts = {s: sum(1 for r in rows if r["status"] == s) for s in ("PASS", "FAIL", "BLOCKED", "NOT RUN")}
    executed = total - counts["NOT RUN"]
    rate = (counts["PASS"] / executed * 100) if executed else 0.0
    p0_fail = [r for r in rows if r["status"] == "FAIL" and r["priority"].startswith("P0")]

    wb = Workbook()

    # ── Summary ──────────────────────────────────────────────────────────────
    ws = wb.active
    ws.title = "Summary"
    ws["A1"] = "REFlect AI - Test Execution Report"
    ws["A1"].font = TITLE_FONT
    ws["A2"] = f"Generated {datetime.now():%Y-%m-%d %H:%M}"
    ws["A2"].font = Font(italic=True, color="666666")

    verdict = ("GO - no blocking defects" if not p0_fail and counts["FAIL"] == 0 else
               "NO-GO - P0 defect open" if p0_fail else
               f"REVIEW - {counts['FAIL']} non-blocking failure(s)")
    ws["A4"] = "Demo readiness"
    ws["B4"] = verdict
    ws["A4"].font = Font(bold=True, size=12)
    ws["B4"].font = Font(bold=True, size=12,
                         color="006100" if verdict.startswith("GO") else "9C0006")

    ws["A6"] = "Metric"; ws["B6"] = "Value"
    _style_header(ws, 6)
    stats = [
        ("Total test cases", total),
        ("Executed", executed),
        ("Passed", counts["PASS"]),
        ("Failed", counts["FAIL"]),
        ("Blocked", counts["BLOCKED"]),
        ("Not run", counts["NOT RUN"]),
        ("Pass rate", f"{rate:.1f}%"),
        ("P0 failures", len(p0_fail)),
    ]
    for i, (k, v) in enumerate(stats, start=7):
        ws.cell(row=i, column=1, value=k).font = Font(bold=True)
        ws.cell(row=i, column=2, value=v)

    r0 = 7 + len(stats) + 1
    ws.cell(row=r0, column=1, value="Layer"); ws.cell(row=r0, column=2, value="Total")
    ws.cell(row=r0, column=3, value="Pass"); ws.cell(row=r0, column=4, value="Fail")
    _style_header(ws, r0)
    for i, layer in enumerate(["Unit", "Integration", "Functional", "Non-functional"], start=r0 + 1):
        lr = [r for r in rows if r["layer"] == layer]
        ws.cell(row=i, column=1, value=layer)
        ws.cell(row=i, column=2, value=len(lr))
        ws.cell(row=i, column=3, value=sum(1 for r in lr if r["status"] == "PASS"))
        ws.cell(row=i, column=4, value=sum(1 for r in lr if r["status"] == "FAIL"))

    r1 = r0 + 6
    ws.cell(row=r1, column=1, value="Priority"); ws.cell(row=r1, column=2, value="Total")
    ws.cell(row=r1, column=3, value="Pass"); ws.cell(row=r1, column=4, value="Fail")
    _style_header(ws, r1)
    for i, prio in enumerate(sorted({r["priority"] for r in rows}), start=r1 + 1):
        pr = [r for r in rows if r["priority"] == prio]
        ws.cell(row=i, column=1, value=prio)
        ws.cell(row=i, column=2, value=len(pr))
        ws.cell(row=i, column=3, value=sum(1 for r in pr if r["status"] == "PASS"))
        ws.cell(row=i, column=4, value=sum(1 for r in pr if r["status"] == "FAIL"))
    _widths(ws, [34, 46, 12, 12])

    # ── Test Cases ───────────────────────────────────────────────────────────
    ws = wb.create_sheet("Test Cases")
    headers = ["Test ID", "Layer", "Module", "Title", "Priority", "Severity",
               "Precondition", "Test Steps", "Expected Result", "Actual Result",
               "Status", "Duration (s)", "Failure Detail", "Automated Test"]
    ws.append(headers)
    _style_header(ws)
    for r in rows:
        ws.append([r["id"], r["layer"], r["module"], r["title"], r["priority"], r["severity"],
                   r["precondition"], r["steps"], r["expected"], r["actual"],
                   r["status"], r["duration"], r["detail"], r["test"]])
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            cell.border = THIN
        st = row[10].value
        row[10].fill = STATUS_FILL.get(st, STATUS_FILL["NOT RUN"])
        row[10].font = STATUS_FONT.get(st, STATUS_FONT["NOT RUN"])
        row[10].alignment = Alignment(horizontal="center", vertical="center")
        if row[4].value in PRIO_FONT:
            row[4].font = PRIO_FONT[row[4].value]
    _widths(ws, [11, 14, 18, 44, 13, 10, 34, 44, 48, 40, 11, 12, 50, 34])
    ws.auto_filter.ref = f"A1:N{ws.max_row}"

    # ── Defects ──────────────────────────────────────────────────────────────
    ws = wb.create_sheet("Defects")
    ws.append(["Defect ID", "Test ID", "Priority", "Severity", "Module",
               "Summary", "Expected", "Actual / Error", "Status"])
    _style_header(ws)
    bad = sorted([r for r in rows if r["status"] in ("FAIL", "BLOCKED")],
                 key=lambda r: r["priority"])
    if bad:
        for i, r in enumerate(bad, start=1):
            ws.append([f"DEF-{i:03d}", r["id"], r["priority"], r["severity"], r["module"],
                       r["title"], r["expected"], (r["actual"] or "") + "\n" + r["detail"], "Open"])
    else:
        ws.append(["-", "-", "-", "-", "-",
                   "No defects open. All executed cases passed.", "-", "-", "Closed"])
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            cell.border = THIN
    _widths(ws, [11, 11, 13, 10, 18, 46, 46, 60, 10])

    # ── Demo Checks ──────────────────────────────────────────────────────────
    ws = wb.create_sheet("Demo Checks")
    ws.append(["#", "Manual pre-demo check", "Why automation cannot cover it", "Done"])
    _style_header(ws)
    manual = [
        ("Sign in with your real ORCID account end to end",
         "Third-party OAuth cannot (and should not) be script-driven; the suite tests up to the auth boundary only."),
        ("Run one full analyse -> compose on the paper you will demo, and read the summary",
         "Narrative quality is a human judgement; automation checks length, format and grounding only."),
        ("Confirm the paper you plan to demo returns GitHub / patent / policy evidence",
         "Evidence yield depends on the specific paper and live upstream APIs."),
        ("Check the ngrok tunnel URL still matches the one baked into the frontend build",
         "The tunnel URL changes if ngrok restarts; the frontend has it compiled in at build time."),
        ("Have a backup: screenshots or a recording of a successful run",
         "Upstream APIs (Semantic Scholar in particular) are rate-limited and outside your control."),
        ("Confirm the Cloudflare Pages deploy is current",
         "The deploy workflow was not firing as of the last check; the live bundle may lag main."),
    ]
    for i, (check, why) in enumerate(manual, start=1):
        ws.append([i, check, why, ""])
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            cell.border = THIN
    _widths(ws, [5, 62, 70, 8])

    wb.save(OUT)
    print(f"[qa] workbook -> {OUT}")
    print(f"[qa] {counts['PASS']}/{executed} passed ({rate:.1f}%) | verdict: {verdict}")
    return 1 if p0_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
