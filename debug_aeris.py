"""
AERIS Diagnostic Script
Run this to check what's working and what's missing.
Usage: python debug_aeris.py
"""
import sys
print(f"Python: {sys.version}")
print("-" * 50)

# 1. WebSockets (for browser UI)
try:
    import websockets
    print(f"✅ websockets: {websockets.__version__}")
except ImportError:
    print("❌ websockets: NOT INSTALLED  →  pip install websockets")

# 2. CustomTkinter
try:
    import customtkinter
    print(f"✅ customtkinter: {customtkinter.__version__}")
except ImportError:
    print("❌ customtkinter: NOT INSTALLED  →  pip install customtkinter")

# 3. SpeechRecognition
try:
    import speech_recognition as sr
    print(f"✅ SpeechRecognition: {sr.__version__}")
except ImportError:
    print("❌ SpeechRecognition: NOT INSTALLED  →  pip install SpeechRecognition")

# 4. PyAudio (mic access)
try:
    import pyaudio
    pa = pyaudio.PyAudio()
    count = pa.get_device_count()
    pa.terminate()
    print(f"✅ pyaudio: installed — {count} audio device(s) found")
except ImportError:
    print("❌ pyaudio: NOT INSTALLED  →  see fix below")
except Exception as e:
    print(f"⚠️  pyaudio: installed but mic error: {e}")

# 5. pyttsx3 (TTS)
try:
    import pyttsx3
    print("✅ pyttsx3: installed")
except ImportError:
    print("❌ pyttsx3: NOT INSTALLED  →  pip install pyttsx3")

# 6. pystray (system tray)
try:
    import pystray
    print("✅ pystray: installed")
except ImportError:
    print("❌ pystray: NOT INSTALLED  →  pip install pystray")

# 7. Microphone test
print("\n--- Microphone Test ---")
try:
    import speech_recognition as sr
    import pyaudio
    r  = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        print("🎤 Mic opened — calibrating (2s)...")
        r.adjust_for_ambient_noise(source, duration=2)
        print(f"✅ Mic OK — energy threshold: {r.energy_threshold:.0f}")
        print("   SAY SOMETHING NOW (5-second window)...")
        audio = r.listen(source, timeout=5, phrase_time_limit=5)
    text = r.recognize_google(audio)
    print(f"✅ STT heard: '{text}'")
except ImportError:
    print("❌ Cannot test mic — pyaudio or SpeechRecognition missing")
except sr.WaitTimeoutError:
    print("⚠️  No speech detected in 5s — mic is open but silent")
except sr.UnknownValueError:
    print("⚠️  Mic works but STT couldn't understand — try again")
except sr.RequestError as e:
    print(f"⚠️  STT request failed (internet needed for Google STT): {e}")
except Exception as e:
    print(f"❌ Mic error: {e}")

print("\n--- Server Test ---")
try:
    import requests
    r = requests.get("http://localhost:8000/status", timeout=3)
    print(f"✅ AERIS server: online (status {r.status_code})")
except Exception as e:
    print(f"❌ AERIS server: not reachable — {e}")
    print("   Start it with: uvicorn main:app --host 0.0.0.0 --port 8000")

print("\nDone.")
