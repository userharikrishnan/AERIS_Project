"""
AERIS Voice Engine — v2.1
==========================
Key fixes from v2.0:
  1. Persistent microphone context — mic is opened ONCE and kept alive.
     The old design re-opened `sr.Microphone()` on every loop frame,
     causing a 500ms–1s cold-start penalty that swallowed wake-word audio.
  2. Wake-word cooldown — 1.5 s dead-band after waking to prevent
     double-triggering on the tail of the utterance.
  3. STT fallback now gets a fresh listen() call on the same open stream,
     so it doesn't race against re-initialization.
  4. Clean separation: wake phase / command phase use the same stream.

Callbacks (called from background threads — must be thread-safe):
  on_hotword()           → hotword detected
  on_command(text)       → full command text recognised
  on_state(state_str)    → state change hint for the UI
  on_transcript(text)    → interim transcript for display
"""
import time
import threading
import queue
import logging
from models.wakeword import WakeWordDetector

logger = logging.getLogger(__name__)

# Broad variants to catch Google STT misinterpretations of "Aeris"
HOTWORD_VARIANTS = [
    "hey aeris", "hey aris", "hey iris", "hey ares", "hey eris",
    "hey areas", "hey aeros", "hey eric", "hey erris", "hey heris",
    "hey aras", "hey eros", "hey heiress", "aeris", "hey harris",
    "hay aeris", "a aeris", "hey area", "hi aeris",
]
# Standalone wake words if STT cuts off the second word
EXACT_WAKE_WORDS = ["hey", "hello", "hi", "wake up", "system online"]

# How long (seconds) to ignore further wake detections after one fires
_WAKE_COOLDOWN = 1.5

# How long (seconds) to listen for calibration on startup
_CALIBRATION_DURATION = 1.5


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

        # ML wake-word engine
        self.wakeword_detector = WakeWordDetector()

        self._tts_queue  = queue.Queue()
        self._stop_event = threading.Event()

        # Timestamp of last successful wake detection
        self._last_wake_ts = 0.0

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

    # ── Public ────────────────────────────────────────────────

    def speak(self, text: str):
        """Queue text for TTS playback. Non-blocking."""
        if self.tts_enabled and text:
            self._tts_queue.put(text)

    def stop(self):
        self._stop_event.set()
        self._tts_queue.put(None)   # unblock TTS worker

    def set_active(self, val: bool):
        self.active = val

    # ── TTS worker ────────────────────────────────────────────

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

    # ── STT / listen loop ─────────────────────────────────────

    def _listen_loop(self):
        """
        Persistent microphone loop.

        The mic is opened ONCE at the top of this method and kept alive for
        the entire session.  We call recognizer.listen() on the persistent
        source rather than re-creating a Microphone context every frame —
        this eliminates the cold-start penalty that was causing missed
        wake-word detections.
        """
        try:
            import speech_recognition as sr
        except ImportError:
            logger.warning("[VoiceEngine] SpeechRecognition not installed — mic disabled")
            return

        recognizer = sr.Recognizer()
        recognizer.energy_threshold        = 300
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold          = 0.7   # slightly snappier than 0.8

        # ── One-time calibration ──────────────────────────────
        try:
            with sr.Microphone() as source:
                logger.info("[VoiceEngine] Calibrating mic...")
                recognizer.adjust_for_ambient_noise(source, duration=_CALIBRATION_DURATION)
                logger.info("[VoiceEngine] Mic calibrated ✓")
        except Exception as e:
            logger.warning(f"[VoiceEngine] Mic calibration failed: {e}")

        # ── Persistent mic context ────────────────────────────
        try:
            mic = sr.Microphone()
        except Exception as e:
            logger.error(f"[VoiceEngine] Cannot open microphone: {e}")
            return

        with mic as source:
            logger.info("[VoiceEngine] Persistent mic stream open ✓")

            while not self._stop_event.is_set():
                if self.mic_muted:
                    time.sleep(0.3)
                    continue

                # ── Listen for audio ─────────────────────────
                try:
                    audio = recognizer.listen(
                        source,
                        timeout=5,           # wait up to 5 s for speech to start
                        phrase_time_limit=8, # max 8 s per utterance
                    )
                except sr.WaitTimeoutError:
                    continue
                except Exception as e:
                    if not self._stop_event.is_set():
                        logger.warning(f"[VoiceEngine] listen() error: {e}")
                    time.sleep(0.1)
                    continue

                # ── WAKE-WORD GATE ────────────────────────────
                if not self.active:
                    is_match = False

                    # 1. Custom Neural Network (100% Offline PyTorch)
                    if self.wakeword_detector.is_active():
                        try:
                            raw_bytes = audio.get_raw_data(
                                convert_rate=16000, convert_width=2
                            )
                            if self.wakeword_detector.predict(raw_bytes):
                                is_match = True
                                logger.info("[VoiceEngine] PyTorch wake-word model fired ✓")
                        except Exception as e:
                            logger.error(f"[VoiceEngine] WakeWord ML error: {e}")

                    # 2. STT text-matching fallback (always try — mic is already captured)
                    if not is_match:
                        try:
                            stt_text = recognizer.recognize_google(audio).lower().strip()
                            logger.debug(f"[STT Wake Fallback] '{stt_text}'")
                            if (
                                any(v in stt_text for v in HOTWORD_VARIANTS)
                                or stt_text in EXACT_WAKE_WORDS
                            ):
                                is_match = True
                                logger.info(f"[VoiceEngine] STT wake-word match: '{stt_text}'")
                        except Exception:
                            pass

                    if is_match:
                        now = time.time()
                        # Cooldown guard — ignore rapid re-triggers
                        if now - self._last_wake_ts < _WAKE_COOLDOWN:
                            logger.debug("[VoiceEngine] Wake ignored — cooldown active")
                            continue
                        self._last_wake_ts = now
                        logger.info("[VoiceEngine] Woke up ✓")
                        self.active = True
                        self.on_hotword()

                    continue   # stay in wake-word phase

                # ── COMMAND PHASE ─────────────────────────────
                try:
                    text = recognizer.recognize_google(audio).lower().strip()
                    logger.debug(f"[STT Command] '{text}'")
                except sr.UnknownValueError:
                    continue
                except sr.RequestError as e:
                    logger.warning(f"[STT] Request error: {e}")
                    continue

                if not text:
                    continue

                self.on_transcript(text)
                self.on_command(text)
