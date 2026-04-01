"""
AERIS Voice Engine
==================
Handles:
  - Continuous mic listening via SpeechRecognition
  - Always-on hotword detection ("hey aeris")
  - Google STT for command recognition
  - Windows TTS via pyttsx3 (runs in its own thread)

Callbacks (called from background threads — must be thread-safe):
  on_hotword()           → hotword detected
  on_command(text)       → full command text recognised
  on_state(state_str)    → state change hint for the UI
  on_transcript(text)    → interim transcript for display
"""
import threading
import queue
import logging

logger = logging.getLogger(__name__)

# Broad variants to catch Google STT misinterpretations of "Aeris"
HOTWORD_VARIANTS = [
    "hey aeris", "hey aris", "hey iris", "hey ares", "hey eris", 
    "hey areas", "hey aeros", "hey eric", "hey erris", "hey heris", 
    "hey aras", "hey eros", "hey heiress", "aeris", "hey harris"
]
# Standalone wake words if STT cuts off the second word
EXACT_WAKE_WORDS = ["hey", "hello", "hi", "wake up", "system online"]

class VoiceEngine:
    def __init__(
        self,
        on_hotword=None,
        on_command=None,
        on_state=None,
        on_transcript=None,
    ):
        self.on_hotword    = on_hotword    or (lambda: None)
        self.on_command    = on_command    or (lambda t: None)
        self.on_state      = on_state      or (lambda s: None)
        self.on_transcript = on_transcript or (lambda t: None)

        self.active      = False   # has hotword been said?
        self.mic_muted   = False
        self.tts_enabled = True

        self._tts_queue  = queue.Queue()
        self._stop_event = threading.Event()

        # TTS thread (pyttsx3 must own its engine in one thread)
        self._tts_thread = threading.Thread(
            target=self._tts_worker, daemon=True, name="aeris-tts"
        )
        self._tts_thread.start()

        # STT / listen thread
        self._listen_thread = threading.Thread(
            target=self._listen_loop, daemon=True, name="aeris-stt"
        )
        self._listen_thread.start()

    # ── Public ────────────────────────────────────────────────────

    def speak(self, text: str):
        """Queue text for TTS playback. Non-blocking."""
        if self.tts_enabled and text:
            self._tts_queue.put(text)

    def stop(self):
        self._stop_event.set()
        self._tts_queue.put(None)   # unblock TTS worker

    def set_active(self, val: bool):
        self.active = val

    # ── TTS worker ────────────────────────────────────────────────

    def _tts_worker(self):
        """pyttsx3 engine runs exclusively in this thread."""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 168)
            engine.setProperty("volume", 1.0)

            # Pick a good Windows voice
            voices = engine.getProperty("voices")
            preferred = next(
                (v for v in voices if any(
                    kw in v.name.lower()
                    for kw in ("david", "mark", "george", "male", "zira")
                )),
                voices[0] if voices else None,
            )
            if preferred:
                engine.setProperty("voice", preferred.id)

            while not self._stop_event.is_set():
                text = self._tts_queue.get()
                if text is None:
                    break
                if self.tts_enabled:
                    self.on_state("speaking")
                    engine.say(text)
                    engine.runAndWait()
                    self.on_state("listening" if self.active else "idle")

        except ImportError:
            logger.warning("[VoiceEngine] pyttsx3 not installed — TTS disabled")
        except Exception as e:
            logger.error(f"[VoiceEngine] TTS error: {e}")

    # ── STT / listen loop ─────────────────────────────────────────

    def _listen_loop(self):
        try:
            import speech_recognition as sr
        except ImportError:
            logger.warning("[VoiceEngine] SpeechRecognition not installed — mic disabled")
            return

        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 300
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 0.8

        # Calibrate once
        try:
            with sr.Microphone() as source:
                logger.info("[VoiceEngine] Calibrating mic...")
                recognizer.adjust_for_ambient_noise(source, duration=1.5)
                logger.info("[VoiceEngine] Mic ready")
        except Exception as e:
            logger.warning(f"[VoiceEngine] Mic calibration failed: {e}")

        while not self._stop_event.is_set():
            if self.mic_muted:
                threading.Event().wait(0.4)
                continue

            try:
                with sr.Microphone() as source:
                    try:
                        audio = recognizer.listen(
                            source, timeout=3, phrase_time_limit=10
                        )
                    except sr.WaitTimeoutError:
                        continue

                try:
                    text = recognizer.recognize_google(audio).lower().strip()
                    logger.debug(f"[STT] '{text}'")
                except sr.UnknownValueError:
                    continue
                except sr.RequestError as e:
                    logger.warning(f"[STT] Request error: {e}")
                    continue

                # ── Hotword gate ───────────────────────────────────
                if not self.active:
                    is_match = False
                    if any(v in text for v in HOTWORD_VARIANTS):
                        is_match = True
                    elif text in EXACT_WAKE_WORDS:
                        is_match = True

                    if is_match:
                        logger.info(f"[VoiceEngine] Woke up on: '{text}'")
                        self.active = True
                        self.on_hotword()
                    continue


                # ── Live command ───────────────────────────────────
                self.on_transcript(text)
                self.on_command(text)

            except Exception as e:
                if not self._stop_event.is_set():
                    logger.warning(f"[VoiceEngine] Listen error: {e}")
