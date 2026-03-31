"""
AERIS Windows Desktop UI
========================
Built with CustomTkinter for a premium dark holographic look.
Features:
  - Animated orb + ring character (OrbCanvas)
  - Always-on hotword via VoiceEngine
  - System tray integration (pystray)
  - Transcript log with colour-coded entries
  - Connects to backend via HTTP POST /core/input
"""
import tkinter as tk
import threading
import queue
import requests
import time
import logging

import customtkinter as ctk
from ui.orb_canvas import OrbCanvas
from ui.voice_engine import VoiceEngine

logger = logging.getLogger(__name__)

# ── Theme ─────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Colour palette (hex)
C_BG       = "#04060f"
C_BG2      = "#080d1c"
C_PANEL    = "#0a1228"
C_BORDER   = "#1a2a50"
C_CYAN     = "#00cfff"
C_GREEN    = "#00ffb3"
C_AMBER    = "#ffaa00"
C_RED      = "#ff3d6b"
C_WHITE    = "#e8f6ff"
C_DIM      = "#5a7080"

FONT_HUD   = ("Courier New", 10, "bold")
FONT_BODY  = ("Segoe UI", 11)
FONT_LABEL = ("Courier New", 9)
FONT_TITLE = ("Courier New", 14, "bold")

BASE_URL   = "http://localhost:8000"


# ─────────────────────────────────────────────────────────────────
# Tray icon helper
# ─────────────────────────────────────────────────────────────────

def _build_tray_icon(show_cb, quit_cb):
    """Create and return a pystray Icon. Runs in its own thread."""
    try:
        import pystray
        from PIL import Image, ImageDraw

        size = 64
        img  = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        # Background circle
        draw.ellipse([2, 2, 62, 62], fill=(4, 6, 15, 255))
        # Glow ring
        for r_off, alpha in [(0, 255), (2, 160), (4, 80)]:
            draw.ellipse(
                [4 + r_off, 4 + r_off, 60 - r_off, 60 - r_off],
                outline=(0, 207, 255, alpha), width=2,
            )
        # Core
        draw.ellipse([22, 22, 42, 42], fill=(0, 207, 255, 255))
        draw.ellipse([27, 27, 37, 37], fill=(255, 255, 255, 255))

        menu = pystray.Menu(
            pystray.MenuItem("Show AERIS", show_cb, default=True),
            pystray.MenuItem("Quit",       quit_cb),
        )
        return pystray.Icon("AERIS", img, "AERIS", menu)
    except Exception as e:
        logger.warning(f"[Tray] pystray not available: {e}")
        return None


# ─────────────────────────────────────────────────────────────────
# Main App Window
# ─────────────────────────────────────────────────────────────────

class AerisApp:
    def __init__(self):
        self._ui_queue  = queue.Queue()   # thread-safe UI updates
        self._conn_ok   = False
        self._tray_icon = None

        self._build_window()
        self._build_header()
        self._build_character()
        self._build_transcript()
        self._build_controls()

        # Voice engine (starts background threads immediately)
        self._voice = VoiceEngine(
            on_hotword    = self._on_hotword,
            on_command    = self._on_command,
            on_state      = self._queue_state,
            on_transcript = self._on_transcript,
        )

        # System tray
        self._setup_tray()

        # Poll connection status
        self._root.after(1000, self._check_connection)
        # Process UI queue at 60 Hz
        self._root.after(16, self._flush_ui_queue)

    # ── Window build ──────────────────────────────────────────────

    def _build_window(self):
        self._root = ctk.CTk()
        self._root.title("AERIS — Autonomous Intelligence")
        self._root.geometry("480x740")
        self._root.minsize(420, 640)
        self._root.configure(fg_color=C_BG)
        self._root.protocol("WM_DELETE_WINDOW", self._on_close_btn)

        # Keyboard shortcut: Escape → minimise to tray
        self._root.bind("<Escape>", lambda _: self._hide_to_tray())

    def _build_header(self):
        hdr = ctk.CTkFrame(self._root, fg_color=C_BG2, corner_radius=0, height=46)
        hdr.pack(fill="x", side="top")
        hdr.pack_propagate(False)

        # Logo
        ctk.CTkLabel(
            hdr, text="⬡  A E R I S",
            font=FONT_TITLE,
            text_color=C_CYAN,
        ).pack(side="left", padx=14, pady=8)

        # Version chip
        ctk.CTkLabel(
            hdr, text="v0.3 // WIN",
            font=FONT_LABEL,
            text_color=C_DIM,
        ).pack(side="left", padx=4)

        # Connection indicator
        self._conn_dot   = ctk.CTkLabel(hdr, text="●", font=("Segoe UI", 12), text_color=C_RED)
        self._conn_label = ctk.CTkLabel(hdr, text="Connecting…", font=FONT_LABEL, text_color=C_DIM)
        self._conn_dot.pack(side="right", padx=(0, 10))
        self._conn_label.pack(side="right")

    def _build_character(self):
        char_frame = ctk.CTkFrame(self._root, fg_color=C_BG, corner_radius=0)
        char_frame.pack(fill="x", padx=0, pady=(4, 0))

        # Canvas orb
        self._orb = OrbCanvas(char_frame, size=300)
        self._orb.pack(pady=(10, 4))

        # Status row
        status_row = ctk.CTkFrame(char_frame, fg_color=C_BG, corner_radius=0)
        status_row.pack(pady=(0, 6))

        self._status_dot  = ctk.CTkLabel(status_row, text="◎", font=("Segoe UI", 11), text_color=C_DIM)
        self._status_text = ctk.CTkLabel(
            status_row, text='Say "Hey AERIS" to activate',
            font=FONT_LABEL, text_color=C_DIM,
        )
        self._status_dot.pack(side="left", padx=(0, 5))
        self._status_text.pack(side="left")

        # Intent readout
        self._intent_lbl = ctk.CTkLabel(char_frame, text="", font=FONT_LABEL, text_color=C_DIM)
        self._intent_lbl.pack(pady=(0, 4))

    def _build_transcript(self):
        # Panel
        panel = ctk.CTkFrame(self._root, fg_color=C_PANEL, corner_radius=10)
        panel.pack(fill="both", expand=True, padx=10, pady=(4, 4))

        # Header row
        hdr = ctk.CTkFrame(panel, fg_color=C_PANEL, corner_radius=0, height=30)
        hdr.pack(fill="x", padx=8, pady=(6, 0))
        hdr.pack_propagate(False)

        ctk.CTkLabel(hdr, text="INTERACTION LOG", font=FONT_LABEL, text_color=C_DIM).pack(side="left")
        ctk.CTkButton(
            hdr, text="✕ Clear", width=60, height=22,
            font=FONT_LABEL, fg_color="transparent",
            text_color=C_RED, hover_color="#2a0010",
            border_width=1, border_color="#550022",
            command=self._clear_transcript,
        ).pack(side="right")

        # Textbox (tkinter Text inside customtkinter)
        self._log = ctk.CTkTextbox(
            panel,
            fg_color=C_PANEL,
            text_color=C_WHITE,
            font=("Segoe UI", 11),
            corner_radius=0,
            wrap="word",
            state="disabled",
        )
        self._log.pack(fill="both", expand=True, padx=6, pady=(4, 6))

        # Colour tags on the underlying tk.Text widget
        tw = self._log._textbox
        tw.tag_config("role_you",   foreground=C_CYAN,  font=("Courier New", 9, "bold"))
        tw.tag_config("role_aeris", foreground=C_GREEN, font=("Courier New", 9, "bold"))
        tw.tag_config("role_sys",   foreground=C_AMBER, font=("Courier New", 9, "bold"))
        tw.tag_config("role_err",   foreground=C_RED,   font=("Courier New", 9, "bold"))
        tw.tag_config("intent_tag", foreground=C_DIM,   font=("Courier New", 8))
        tw.tag_config("body_you",   foreground="#b0d8f0")
        tw.tag_config("body_aeris", foreground="#b0ffe0")
        tw.tag_config("body_sys",   foreground="#ffe0a0")
        tw.tag_config("body_err",   foreground="#ffb0b0")

    def _build_controls(self):
        bar = ctk.CTkFrame(self._root, fg_color=C_BG2, corner_radius=0, height=60)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)

        btn_kw = dict(width=90, height=38, corner_radius=8, font=FONT_LABEL)

        self._mic_btn = ctk.CTkButton(
            bar, text="🎤  Mic On",
            fg_color="#0a1a30", hover_color="#102040",
            text_color=C_CYAN, border_width=1, border_color=C_BORDER,
            command=self._toggle_mic, **btn_kw,
        )
        self._mic_btn.pack(side="left", padx=8, pady=10)

        self._tts_btn = ctk.CTkButton(
            bar, text="🔊  Voice",
            fg_color="#0a1a30", hover_color="#102040",
            text_color=C_GREEN, border_width=1, border_color=C_BORDER,
            command=self._toggle_tts, **btn_kw,
        )
        self._tts_btn.pack(side="left", padx=4, pady=10)

        ctk.CTkLabel(
            bar, text='⌀  "Hey AERIS"',
            font=FONT_LABEL, text_color=C_DIM,
        ).pack(side="left", padx=10)

        ctk.CTkButton(
            bar, text="🌐  Browser",
            fg_color="#0a1a30", hover_color="#102040",
            text_color=C_DIM, border_width=1, border_color=C_BORDER,
            command=lambda: __import__("webbrowser").open(BASE_URL),
            **btn_kw,
        ).pack(side="right", padx=8, pady=10)

    # ── System Tray ───────────────────────────────────────────────

    def _setup_tray(self):
        icon = _build_tray_icon(
            show_cb=lambda: self._ui_queue.put(("show", None)),
            quit_cb=lambda: self._ui_queue.put(("quit", None)),
        )
        if icon:
            self._tray_icon = icon
            threading.Thread(
                target=icon.run, daemon=True, name="aeris-tray"
            ).start()
        self._root.protocol("WM_DELETE_WINDOW", self._on_close_btn)

    def _hide_to_tray(self):
        self._root.withdraw()

    def _show_from_tray(self):
        self._root.deiconify()
        self._root.lift()
        self._root.focus_force()

    # ── Voice callbacks (all called from non-main threads) ────────

    def _on_hotword(self):
        self._queue_state("greeting")
        greeting = self._random_greeting()
        self._queue_transcript("aeris", greeting, "GREETING")
        self._voice.speak(greeting)

    def _on_command(self, text: str):
        self._queue_transcript("you", text)
        self._queue_state("thinking")

        def _post():
            try:
                r = requests.post(
                    f"{BASE_URL}/core/input",
                    json={"text": text},
                    timeout=15,
                )
                data = r.json()
                response = data.get("response", "")
                intent   = data.get("intent", data.get("mode", ""))
                self._ui_queue.put(("transcript", ("aeris", response, intent)))
                self._ui_queue.put(("intent", intent))
                self._ui_queue.put(("state", "listening" if self._voice.active else "idle"))
                if self._voice.tts_enabled and response:
                    self._voice.speak(response)
            except Exception as e:
                msg = f"Could not reach AERIS server: {e}"
                self._ui_queue.put(("transcript", ("sys", msg, "error")))
                self._ui_queue.put(("state", "offline"))

        threading.Thread(target=_post, daemon=True).start()

    def _on_transcript(self, text: str):
        # Update status to show live transcript
        self._ui_queue.put(("status_text", f"Heard: {text[:50]}"))

    def _queue_state(self, state: str):
        self._ui_queue.put(("state", state))

    def _queue_transcript(self, role: str, text: str, intent: str = ""):
        self._ui_queue.put(("transcript", (role, text, intent)))

    # ── UI queue processor (main thread) ──────────────────────────

    def _flush_ui_queue(self):
        try:
            while True:
                cmd, val = self._ui_queue.get_nowait()
                if cmd == "state":
                    self._apply_state(val)
                elif cmd == "transcript":
                    self._add_entry(*val)
                elif cmd == "status_text":
                    self._status_text.configure(text=val)
                elif cmd == "intent":
                    self._intent_lbl.configure(
                        text=f"◈  {val}" if val else "",
                        text_color=C_DIM,
                    )
                elif cmd == "show":
                    self._show_from_tray()
                elif cmd == "quit":
                    self._quit()
        except queue.Empty:
            pass
        self._root.after(16, self._flush_ui_queue)

    # ── State → visual ────────────────────────────────────────────

    _STATE_TEXT = {
        "idle":      ('Say "Hey AERIS" to activate', C_DIM),
        "listening": ("Listening…",                  C_CYAN),
        "thinking":  ("Processing…",                 C_AMBER),
        "speaking":  ("Speaking…",                   C_GREEN),
        "greeting":  ("AERIS Online",                "#ffffff"),
        "offline":   ("Connection Lost",             C_RED),
    }

    def _apply_state(self, state: str):
        self._orb.set_state(state)
        text, color = self._STATE_TEXT.get(state, ('', C_DIM))
        self._status_text.configure(text=text, text_color=color)

        # Dot colour
        dot_colors = {
            "idle": C_DIM, "listening": C_CYAN, "thinking": C_AMBER,
            "speaking": C_GREEN, "greeting": "#ffffff", "offline": C_RED,
        }
        self._status_dot.configure(text_color=dot_colors.get(state, C_DIM))

    # ── Transcript ────────────────────────────────────────────────

    def _add_entry(self, role: str, text: str, intent: str = ""):
        tw = self._log._textbox
        self._log.configure(state="normal")

        role_map  = {"you": "YOU  ", "aeris": "AERIS", "sys": "SYS  ", "error": "ERR  "}
        role_tag  = {"you": "role_you", "aeris": "role_aeris", "sys": "role_sys", "error": "role_err"}
        body_tag  = {"you": "body_you", "aeris": "body_aeris", "sys": "body_sys",  "error": "body_err"}

        rl  = role_map .get(role, "     ")
        rt  = role_tag .get(role, "role_sys")
        bt  = body_tag .get(role, "body_sys")

        if intent and role == "aeris":
            tw.insert("end", f"[{intent}]\n", "intent_tag")
        tw.insert("end", f"{rl}  ", rt)
        tw.insert("end", f"{text}\n\n", bt)

        self._log.configure(state="disabled")
        tw.see("end")

    def _clear_transcript(self):
        self._log.configure(state="normal")
        self._log._textbox.delete("1.0", "end")
        self._log.configure(state="disabled")
        self._intent_lbl.configure(text="")

    # ── Connection status ─────────────────────────────────────────

    def _check_connection(self):
        def _ping():
            try:
                r = requests.get(f"{BASE_URL}/status", timeout=2)
                ok = r.status_code == 200
            except Exception:
                ok = False
            self._ui_queue.put(("_conn", ok))

        threading.Thread(target=_ping, daemon=True).start()
        self._root.after(5000, self._check_connection)

    # Handle _conn internally
    def _flush_ui_queue(self):
        try:
            while True:
                cmd, val = self._ui_queue.get_nowait()
                if cmd == "state":
                    self._apply_state(val)
                elif cmd == "transcript":
                    self._add_entry(*val)
                elif cmd == "status_text":
                    self._status_text.configure(text=val)
                elif cmd == "intent":
                    self._intent_lbl.configure(
                        text=f"◈  {val}" if val else "", text_color=C_DIM
                    )
                elif cmd == "_conn":
                    self._conn_ok = val
                    if val:
                        self._conn_dot.configure(text_color=C_GREEN)
                        self._conn_label.configure(text="Connected", text_color=C_GREEN)
                    else:
                        self._conn_dot.configure(text_color=C_RED)
                        self._conn_label.configure(text="Offline", text_color=C_RED)
                elif cmd == "show":
                    self._show_from_tray()
                elif cmd == "quit":
                    self._quit()
        except queue.Empty:
            pass
        self._root.after(16, self._flush_ui_queue)

    # ── Controls ──────────────────────────────────────────────────

    def _toggle_mic(self):
        self._voice.mic_muted = not self._voice.mic_muted
        if self._voice.mic_muted:
            self._mic_btn.configure(text="🔇  Muted", text_color=C_RED)
        else:
            self._mic_btn.configure(text="🎤  Mic On", text_color=C_CYAN)

    def _toggle_tts(self):
        self._voice.tts_enabled = not self._voice.tts_enabled
        if not self._voice.tts_enabled:
            self._tts_btn.configure(text="🔇  Muted", text_color=C_RED)
        else:
            self._tts_btn.configure(text="🔊  Voice", text_color=C_GREEN)

    # ── Lifecycle ─────────────────────────────────────────────────

    def _on_close_btn(self):
        """Minimise to tray on window close."""
        if self._tray_icon:
            self._hide_to_tray()
        else:
            self._quit()

    def _quit(self):
        self._voice.stop()
        self._orb.stop()
        if self._tray_icon:
            self._tray_icon.stop()
        self._root.destroy()

    def run(self):
        self._root.mainloop()

    # ── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _random_greeting() -> str:
        import random
        msgs = [
            "AERIS online. Systems nominal. At your service.",
            "Good to hear from you. How can I assist?",
            "All systems online. What do you need?",
            "AERIS ready. Awaiting your command.",
        ]
        return random.choice(msgs)
