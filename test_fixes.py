"""
Quick smoke-test for the 3 AERIS fixes.
Run from the project root with: venv\Scripts\python.exe tmp\test_fixes.py
"""
import sys
import os
sys.path.insert(0, os.path.abspath("."))

print("=" * 60)
print("  AERIS Fix Smoke Tests")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# 1. RULE-BASED CLASSIFIER
# ─────────────────────────────────────────────────────────────
print("\n[1] Rule-based classifier tests")
from services.nlp_processor import NLPProcessor
nlp = NLPProcessor()

test_cases = [
    # (input_text, expected_intent, expected_min_conf, expected_app_or_query)
    ("open chrome",          "OPEN_APP",    0.90, "chrome"),
    ("open edge",            "OPEN_APP",    0.90, "edge"),
    ("open ed",              "OPEN_APP",    0.90, "edge"),     # alias: ed → edge
    ("launch spotify",       "OPEN_APP",    0.90, "spotify"),
    ("close discord",        "CLOSE_APP",   0.90, "discord"),
    ("search for python",    "WEB_SEARCH",  0.90, None),
    ("google the weather",   "WEB_SEARCH",  0.90, None),
    ("go to github.com",     "WEB_NAVIGATE",0.90, None),
    ("take a screenshot",    "SCREENSHOT",  0.90, None),
    ("remember that my name is hari", "MEMORY_STORE", 0.90, None),
    ("who are you",          "IDENTITY_QUERY", 0.90, None),
]

all_pass = True
for text, exp_intent, min_conf, exp_app in test_cases:
    intent = nlp.extract_intent(text)
    passed = (
        intent.type == exp_intent and
        intent.confidence >= min_conf
    )
    # Check entity extraction if expected
    if exp_app and passed:
        actual_app = intent.entities.get("app", "")
        passed = actual_app == exp_app

    status = "✓ PASS" if passed else "✗ FAIL"
    if not passed:
        all_pass = False

    app_info = f"  app={intent.entities.get('app','—')}" if exp_app else ""
    print(f"  {status}  '{text}'")
    print(f"          → intent={intent.type} conf={intent.confidence:.2f}{app_info}")
    if not passed:
        print(f"          EXPECTED intent={exp_intent} conf>={min_conf}" +
              (f" app={exp_app}" if exp_app else ""))

# ─────────────────────────────────────────────────────────────
# 2. LANGUAGE ENGINE — no hallucinations for action intents
# ─────────────────────────────────────────────────────────────
print("\n[2] Language engine — action intents must use templates")
from services.language_engine import LanguageEngine
from models.slm import AerisSLM
from models.tokenizer import Tokenizer
tok = Tokenizer()
slm = AerisSLM(vocab_size=1000, embed_dim=384, num_layers=6,
               num_heads=6, ffn_dim=1536, dropout=0.0)
le = LanguageEngine(slm, tok)

action_tests = [
    ("OPEN_APP",  {"app": "chrome"}, "Opening chrome"),
    ("CLOSE_APP", {"app": "edge"},   "Closing edge"),
    ("WEB_SEARCH",{"query": "python tutorial"}, "Searching"),
    ("SCREENSHOT",{}, "Capturing"),
    ("MEMORY_STORE", {}, "stored"),
]

HALLUCINATIONS = [
    "choose the safer option", "safer option first", "optimize if needed",
    "searching the failure", "searching for the failure", "be the safest",
    "cannot determine", "i am not sure", "i am unable",
]

for intent_type, entities, must_contain in action_tests:
    resp = le.generate_from_text(
        original_text="test command",
        intent_type=intent_type,
        confidence=0.97,
        entities=entities,
    )
    hallucinated = any(h in resp.lower() for h in HALLUCINATIONS)
    has_content  = must_contain.lower() in resp.lower()
    passed = has_content and not hallucinated
    status = "✓ PASS" if passed else "✗ FAIL"
    if not passed:
        all_pass = False
    print(f"  {status}  {intent_type} → '{resp}'")
    if not passed:
        print(f"          Expected to contain '{must_contain}', hallucinated={hallucinated}")

# ─────────────────────────────────────────────────────────────
# 3. WAKE WORD THRESHOLD
# ─────────────────────────────────────────────────────────────
print("\n[3] Wake word threshold check")
from models.wakeword import WakeWordDetector
wd = WakeWordDetector()
# just verify it loads and the threshold is right by inspecting source
import inspect
src = inspect.getsource(wd.predict)
threshold_ok = "0.55" in src and "0.8" not in src.split("0.55")[0].split("confidence >")[-1]
status = "✓ PASS" if threshold_ok else "✗ FAIL (check wakeword.py threshold)"
print(f"  {status}  Threshold = 0.55 confirmed in predict()")

# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
if all_pass:
    print("  ALL TESTS PASSED ✓  — ready to run launcher.py")
else:
    print("  SOME TESTS FAILED — check output above")
print("=" * 60)
