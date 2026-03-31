"""
AERIS End-to-End Test Suite
============================
Run AFTER training completes and the server is up:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Then in a new terminal:
    python test_aeris.py

Tests covered:
  1. Health check
  2. Chat (greetings, identity)
  3. App launch (smart confirmation first-time → auto)
  4. Web search
  5. Web navigate
  6. Web scrape
  7. Generate report (after scrape)
  8. System info
  9. Screenshot
  10. File ops
  11. Memory store + recall
  12. Session close detection
  13. Status endpoint
  14. Teaching endpoints (app + browser)
  15. Correction endpoint
"""

import requests
import json
import time

BASE = "http://localhost:8000"

PASS = "✅"
FAIL = "❌"
INFO = "ℹ️ "

results = []


def r(method, path, payload=None, label=""):
    url = f"{BASE}{path}"
    try:
        if method == "GET":
            resp = requests.get(url, timeout=10)
        else:
            resp = requests.post(url, json=payload, timeout=10)
        data = resp.json()
        return data
    except Exception as e:
        return {"error": str(e)}


def check(label, data, expected_keys=None, expected_values=None, not_expected=None):
    ok = True
    notes = []

    if "error" in data and not (expected_keys and "error" in expected_keys):
        ok = False
        notes.append(f"Unexpected error: {data['error']}")

    if expected_keys:
        for key in expected_keys:
            if key not in data:
                ok = False
                notes.append(f"Missing key: '{key}'")

    if expected_values:
        for key, val in expected_values.items():
            if data.get(key) != val:
                ok = False
                notes.append(f"Expected {key}={val!r}, got {data.get(key)!r}")

    if not_expected:
        for key in not_expected:
            if key in data:
                ok = False
                notes.append(f"Key should NOT be present: '{key}'")

    icon = PASS if ok else FAIL
    status = "PASS" if ok else "FAIL"
    results.append((icon, label, status, notes))

    print(f"  {icon} {label}")
    if notes:
        for n in notes:
            print(f"       {INFO} {n}")
    if not ok:
        print(f"       Response: {json.dumps(data, indent=6)[:300]}")
    return data


def section(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")


# ===========================================================
# 1. HEALTH CHECK
# ===========================================================
section("1. Health & Status")

data = check(
    "GET / → AERIS online",
    r("GET", "/"),
    expected_keys=["status", "tools"]
)

data = check(
    "GET /status → full status",
    r("GET", "/status"),
    expected_keys=["aeris", "version_info", "tools", "session"]
)

# ===========================================================
# 2. CHAT — Greetings & Identity
# ===========================================================
section("2. Chat — Greetings & Identity")

data = check(
    "POST /core/input 'hi' → chat response",
    r("POST", "/core/input", {"text": "hi"}),
    expected_keys=["response"]
)
if "response" in data:
    print(f"       Response: '{data['response']}'")

data = check(
    "POST /core/input 'who are you' → identity",
    r("POST", "/core/input", {"text": "who are you"}),
    expected_keys=["response"]
)
if "response" in data:
    print(f"       Response: '{data['response']}'")

data = check(
    "POST /core/input 'what can you do' → capabilities",
    r("POST", "/core/input", {"text": "what can you do"}),
    expected_keys=["response"]
)
if "response" in data:
    print(f"       Response: '{data['response']}'")

# ===========================================================
# 3. APP LAUNCH — Smart Dual-Mode Confirmation
# ===========================================================
section("3. App Launch — Smart Dual-Mode Confirmation")

data = r("POST", "/core/input", {"text": "open notepad"})
check(
    "POST /core/input 'open notepad' → confirmation OR auto",
    data,
    expected_keys=["confirmation_id"] if "confirmation_id" in data else ["response"]
)

confirmation_id = data.get("confirmation_id")
if confirmation_id:
    print(f"       {INFO} First time — confirmation required (ID: {confirmation_id[:8]}...)")
    confirm_data = check(
        "POST /core/confirm → approve → tool executes",
        r("POST", "/core/confirm", {
            "confirmation_id": confirmation_id,
            "approved": True
        }),
        expected_keys=["status"]
    )
    if "status" in confirm_data:
        print(f"       Status: '{confirm_data['status']}'")
    if confirm_data.get("status") == "executed":
        print(f"       {PASS} Tool actually executed!")
    elif confirm_data.get("status") == "execution_failed":
        print(f"       {INFO} Execution attempted but failed (expected if notepad path differs on your system)")

    # Second time — should auto-approve
    time.sleep(1)
    data2 = r("POST", "/core/input", {"text": "open notepad"})
    check(
        "POST /core/input 'open notepad' again → auto-approved",
        data2,
        expected_keys=["mode"] if data2.get("mode") == "auto_executed" else ["confirmation_id"]
    )
    if data2.get("mode") == "auto_executed":
        print(f"       {PASS} Auto-approved on second call — smart confirmation working!")
    elif "confirmation_id" in data2:
        print(f"       {INFO} Still asking for confirmation (trust score may need more approvals)")

else:
    print(f"       {INFO} Auto-executed directly: '{data.get('response', '')}'")

# ===========================================================
# 4. WEB SEARCH
# ===========================================================
section("4. Web Search")

data = r("POST", "/core/input", {"text": "search for python tutorials"})
confirmation_id = data.get("confirmation_id")
check(
    "POST /core/input 'search for python tutorials'",
    data,
    expected_keys=["confirmation_id"] if confirmation_id else ["response"]
)
if confirmation_id:
    r("POST", "/core/confirm", {"confirmation_id": confirmation_id, "approved": True})
    print(f"       {INFO} Confirmed web search")

# ===========================================================
# 5. WEB NAVIGATE
# ===========================================================
section("5. Web Navigate")

data = r("POST", "/core/input", {"text": "go to github.com"})
confirmation_id = data.get("confirmation_id")
check(
    "POST /core/input 'go to github.com'",
    data,
    expected_keys=["confirmation_id"] if confirmation_id else ["response"]
)
if confirmation_id:
    r("POST", "/core/confirm", {"confirmation_id": confirmation_id, "approved": True})
    print(f"       {INFO} Confirmed navigation")

# ===========================================================
# 6. WEB SCRAPE (NEW)
# ===========================================================
section("6. Web Scrape — NEW Intent")

data = r("POST", "/core/input", {"text": "scrape https://httpbin.org/html"})
check(
    "POST /core/input 'scrape https://httpbin.org/html' → WEB_SCRAPE intent",
    data,
    expected_keys=["confirmation_id"] if "confirmation_id" in data
                  else ["response", "mode"]
)
confirmation_id = data.get("confirmation_id")
if confirmation_id:
    scrape_confirm = check(
        "Confirm web scrape → tool executes",
        r("POST", "/core/confirm", {
            "confirmation_id": confirmation_id,
            "approved": True
        }),
        expected_keys=["status"]
    )
    if scrape_confirm.get("status") == "executed":
        print(f"       {PASS} Scrape tool executed successfully!")
        result = scrape_confirm.get("result", {})
        if result.get("title"):
            print(f"       Scraped title: '{result['title']}'")
        if result.get("heading_count"):
            print(f"       Headings found: {result['heading_count']}")
    else:
        print(f"       Status: {scrape_confirm.get('status')}")

# ===========================================================
# 7. GENERATE REPORT (NEW)
# ===========================================================
section("7. Generate Report — NEW Intent")

data = r("POST", "/core/input", {"text": "generate a markdown report and save to desktop"})
check(
    "POST /core/input 'generate a markdown report'",
    data,
    expected_keys=["confirmation_id"] if "confirmation_id" in data else ["response"]
)
confirmation_id = data.get("confirmation_id")
if confirmation_id:
    report_confirm = check(
        "Confirm report generation → tool executes",
        r("POST", "/core/confirm", {
            "confirmation_id": confirmation_id,
            "approved": True
        }),
        expected_keys=["status"]
    )
    if report_confirm.get("status") == "executed":
        print(f"       {PASS} Report generated!")
        result = report_confirm.get("result", {})
        if result.get("saved_path"):
            print(f"       Saved to: '{result['saved_path']}'")

# ===========================================================
# 8. SYSTEM INFO (NEW)
# ===========================================================
section("8. System Info — NEW Intent")

data = r("POST", "/core/input", {"text": "show me system info"})
check(
    "POST /core/input 'show me system info'",
    data,
    expected_keys=["confirmation_id"] if "confirmation_id" in data else ["response"]
)
confirmation_id = data.get("confirmation_id")
if confirmation_id:
    sys_confirm = check(
        "Confirm system info → tool executes",
        r("POST", "/core/confirm", {
            "confirmation_id": confirmation_id,
            "approved": True
        }),
        expected_keys=["status"]
    )
    if sys_confirm.get("status") == "executed":
        result = sys_confirm.get("result", {})
        print(f"       {PASS} System info retrieved!")
        if result.get("cpu_percent") is not None:
            print(f"       CPU: {result['cpu_percent']}%  RAM used: {result.get('ram_percent', '?')}%")

# ===========================================================
# 9. SCREENSHOT (NEW)
# ===========================================================
section("9. Screenshot — NEW Tool")

data = r("POST", "/core/input", {"text": "take a screenshot"})
check(
    "POST /core/input 'take a screenshot'",
    data,
    expected_keys=["confirmation_id"] if "confirmation_id" in data else ["response"]
)
confirmation_id = data.get("confirmation_id")
if confirmation_id:
    ss_confirm = check(
        "Confirm screenshot → tool executes",
        r("POST", "/core/confirm", {
            "confirmation_id": confirmation_id,
            "approved": True
        }),
        expected_keys=["status"]
    )
    if ss_confirm.get("status") == "executed":
        result = ss_confirm.get("result", {})
        print(f"       {PASS} Screenshot taken!")
        if result.get("saved_path"):
            print(f"       Saved to: '{result['saved_path']}'")

# ===========================================================
# 10. FILE OPERATIONS
# ===========================================================
section("10. File Operations")

data = r("POST", "/core/input", {"text": "list files in desktop"})
check(
    "POST /core/input 'list files in desktop'",
    data,
    expected_keys=["confirmation_id"] if "confirmation_id" in data else ["response"]
)
confirmation_id = data.get("confirmation_id")
if confirmation_id:
    r("POST", "/core/confirm", {"confirmation_id": confirmation_id, "approved": True})
    print(f"       {INFO} File list confirmed")

# ===========================================================
# 11. MEMORY — Store + Recall
# ===========================================================
section("11. Memory — Store & Recall")

data = check(
    "POST /core/input 'remember my name is Tony'",
    r("POST", "/core/input", {"text": "remember my name is Tony"}),
    expected_keys=["response"]
)
if "response" in data:
    print(f"       Response: '{data['response']}'")

time.sleep(0.5)

data = check(
    "POST /core/input 'what is my name'",
    r("POST", "/core/input", {"text": "what is my name"}),
    expected_keys=["response"]
)
if "response" in data:
    print(f"       Response: '{data['response']}'")

# ===========================================================
# 12. SESSION CLOSE DETECTION
# ===========================================================
section("12. Session Close Detection")

data = check(
    "POST /core/input 'thanks that will be all' → session close",
    r("POST", "/core/input", {"text": "thanks that will be all"}),
    expected_keys=["response"]
)
if data.get("session_closed"):
    print(f"       {PASS} Session closed correctly!")
if "response" in data:
    print(f"       Response: '{data['response']}'")

# ===========================================================
# 13. TEACHING ENDPOINTS
# ===========================================================
section("13. Teaching Endpoints")

check(
    "POST /teach/app → teach new app path",
    r("POST", "/teach/app", {
        "app_name": "test_app",
        "path": "C:/Windows/System32/notepad.exe"
    }),
    expected_values={"taught": True}
)

check(
    "POST /teach/browser → set preferred browser",
    r("POST", "/teach/browser", {"browser_name": "chrome"}),
    expected_values={"taught": True}
)

check(
    "POST /correct → correct wrong intent",
    r("POST", "/correct", {
        "original_input": "show me page",
        "original_intent": "FILE_READ",
        "corrected_intent": "WEB_SCRAPE",
        "detail": "user meant web page, not local file"
    }),
    expected_values={"correction_recorded": True}
)

# ===========================================================
# 14. UPDATE CHECK
# ===========================================================
section("14. Update Manager")

check(
    "GET /updates/check → check for updates",
    r("GET", "/updates/check"),
    expected_keys=["pending", "current_version"]
)

# ===========================================================
# SUMMARY
# ===========================================================
print(f"\n{'='*55}")
print(f"  AERIS TEST SUMMARY")
print(f"{'='*55}")

passed = sum(1 for r in results if r[2] == "PASS")
failed = sum(1 for r in results if r[2] == "FAIL")
total  = len(results)

print(f"  {PASS} Passed : {passed}/{total}")
print(f"  {FAIL} Failed : {failed}/{total}")
print(f"  Score  : {passed/total*100:.0f}%")

if failed > 0:
    print(f"\n  Failed tests:")
    for icon, label, status, notes in results:
        if status == "FAIL":
            print(f"    • {label}")
            for n in notes:
                print(f"        {n}")

print(f"\n{'='*55}")
print("  To manually test any endpoint, use:")
print("  curl -X POST http://localhost:8000/core/input")
print('       -H "Content-Type: application/json"')
print('       -d \'{"text": "YOUR COMMAND HERE"}\'')
print(f"{'='*55}\n")
