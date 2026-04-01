"""
AERIS Hotword Debug — Reveals exactly what Google STT hears.
Run this, then say "hey aeris" (or play it from your phone).
It will print EVERY word it hears so you can see what STT transcribes.

Usage:  python debug_hotword.py
"""
import speech_recognition as sr
import sys

HOTWORD = "hey aeris"

# All reasonable mis-transcriptions of "hey aeris"
FUZZY_VARIANTS = [
    "hey aeris", "hey aris", "hey iris", "hey ares",
    "hey eris", "hey aerys", "hey aeries", "hey aris",
    "a aeris", "hey areas", "hey aeros", "hey a. r. i. s.",
    "hey arris", "hey ayris", "aeris", "hay aeris",
    "hey eric", "hey erris", "hey heris", "hey aras",
    "hey eros", "hey paris", "hey heiress",
]


def fuzzy_match(text: str) -> bool:
    text = text.lower().strip()
    for v in FUZZY_VARIANTS:
        if v in text:
            return True
    # Also check if 'aeris' appears anywhere
    if "aeris" in text:
        return True
    return False


def main():
    print("=" * 60)
    print("  AERIS HOTWORD DIAGNOSTIC")
    print("  Say 'Hey AERIS' — I'll show exactly what STT hears")
    print("=" * 60)

    r = sr.Recognizer()
    r.energy_threshold = 300
    r.dynamic_energy_threshold = True
    r.pause_threshold = 0.8

    with sr.Microphone() as mic:
        print("\n🎤 Calibrating mic (2s)...")
        r.adjust_for_ambient_noise(mic, duration=2)
        print(f"✅ Ready — energy threshold: {r.energy_threshold:.0f}")
        print("\n" + "-" * 60)
        print("SPEAK NOW. I will print everything I hear.")
        print("Press Ctrl+C to exit.")
        print("-" * 60 + "\n")

        round_num = 0
        while True:
            round_num += 1
            try:
                print(f"[Round {round_num}] Listening (up to 8s)...", end=" ", flush=True)
                audio = r.listen(mic, timeout=8, phrase_time_limit=10)
                print("Processing...", end=" ", flush=True)

                try:
                    text = r.recognize_google(audio)
                    text_lower = text.lower().strip()
                    print(f"\n  📝 STT heard: \"{text}\"")

                    if fuzzy_match(text_lower):
                        print(f"  ✅ HOTWORD MATCH! \"{text_lower}\" matches a variant")
                    else:
                        print(f"  ❌ No hotword match. Looking for: 'hey aeris'")
                        print(f"     (exact text was: \"{text_lower}\")")

                except sr.UnknownValueError:
                    print("\n  ⚠️  STT could not understand the audio")
                    print("     (heard noise but couldn't make out words)")

                except sr.RequestError as e:
                    print(f"\n  ❌ Google STT request failed: {e}")
                    print("     Check your internet connection")

                print()

            except sr.WaitTimeoutError:
                print("(no speech detected)\n")
            except KeyboardInterrupt:
                print("\n\nDone.")
                sys.exit(0)


if __name__ == "__main__":
    main()
