"""
AERIS Dynamic App Registry
Discovers installed applications from the Windows filesystem and registry.
Replaces the hardcoded APP_MAP with a learnable, expandable registry.

Sources scanned:
1. Windows Registry (HKCU and HKLM software entries)
2. Start Menu shortcuts (.lnk files)
3. Common installation paths (Program Files, AppData, PATH)
4. User-taught mappings (stored in SQLite)

Learning:
- When user corrects a wrong app → saves user mapping
- When user answers "which browser?" → saves app preference
- Mappings persist across sessions and take priority over discovered ones
"""

import os
import sys
import sqlite3
import logging
import time
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)

# Common paths to scan for executables
SCAN_PATHS = [
    r"C:\Program Files",
    r"C:\Program Files (x86)",
    os.path.expandvars(r"%APPDATA%\Microsoft\Windows\Start Menu\Programs"),
    os.path.expandvars(r"%PROGRAMDATA%\Microsoft\Windows\Start Menu\Programs"),
    os.path.expandvars(r"%LOCALAPPDATA%"),
]

# Well-known app name aliases (normalized name → executable patterns)
KNOWN_ALIASES: Dict[str, List[str]] = {
    "chrome": ["chrome.exe", "google chrome"],
    "firefox": ["firefox.exe"],
    "edge": ["msedge.exe", "microsoft edge"],
    "brave": ["brave.exe"],
    "opera": ["opera.exe"],
    "notepad": ["notepad.exe"],
    "notepad++": ["notepad++.exe"],
    "vscode": ["code.exe", "visual studio code"],
    "vs code": ["code.exe"],
    "visual studio code": ["code.exe"],
    "word": ["winword.exe", "microsoft word"],
    "excel": ["excel.exe", "microsoft excel"],
    "powerpoint": ["powerpnt.exe"],
    "outlook": ["outlook.exe"],
    "teams": ["teams.exe", "microsoft teams"],
    "discord": ["discord.exe"],
    "slack": ["slack.exe"],
    "zoom": ["zoom.exe"],
    "spotify": ["spotify.exe"],
    "steam": ["steam.exe"],
    "vlc": ["vlc.exe"],
    "obs": ["obs64.exe", "obs.exe"],
    "calculator": ["calc.exe"],
    "calc": ["calc.exe"],
    "cmd": ["cmd.exe"],
    "powershell": ["powershell.exe"],
    "terminal": ["wt.exe", "WindowsTerminal.exe"],
    "explorer": ["explorer.exe"],
    "paint": ["mspaint.exe"],
    "paint.net": ["PaintDotNet.exe"],
    "pycharm": ["pycharm64.exe"],
    "intellij": ["idea64.exe"],
    "android studio": ["studio64.exe"],
    "postman": ["postman.exe"],
    "docker": ["docker desktop.exe", "docker.exe"],
    "task manager": ["taskmgr.exe"],
    "control panel": ["control.exe"],
    "settings": ["ms-settings:"],
    "wordpad": ["wordpad.exe"],
}


class AppRegistry:
    """
    Dynamic application registry that discovers and learns app paths.
    """

    def __init__(self, db_path: str = "db/apps.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._cache: Dict[str, str] = {}  # In-memory LRU cache
        self._init_db()
        self._load_cache()

    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS apps (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE,
                path TEXT,
                source TEXT,
                confidence REAL DEFAULT 1.0,
                use_count INTEGER DEFAULT 0,
                last_used REAL,
                user_taught INTEGER DEFAULT 0
            )
        """)
        self.conn.commit()
        self._seed_system_apps()

    def _seed_system_apps(self):
        """Seed with known-safe system apps that always exist on Windows."""
        system_apps = [
            ("notepad", "notepad.exe", "system"),
            ("calculator", "calc.exe", "system"),
            ("calc", "calc.exe", "system"),
            ("cmd", "cmd.exe", "system"),
            ("powershell", "powershell.exe", "system"),
            ("explorer", "explorer.exe", "system"),
            ("paint", "mspaint.exe", "system"),
            ("wordpad", "write.exe", "system"),
            ("task manager", "taskmgr.exe", "system"),
            ("control panel", "control.exe", "system"),
            ("snipping tool", "SnippingTool.exe", "system"),
            ("magnifier", "magnify.exe", "system"),
            ("narrator", "narrator.exe", "system"),
        ]

        for name, path, source in system_apps:
            try:
                self.conn.execute("""
                    INSERT OR IGNORE INTO apps (name, path, source, confidence, user_taught)
                    VALUES (?, ?, ?, 1.0, 0)
                """, (name, path, source))
            except Exception:
                pass
        self.conn.commit()

    def _load_cache(self):
        """Load all app mappings into memory for fast lookup."""
        cursor = self.conn.execute("SELECT name, path FROM apps ORDER BY user_taught DESC, use_count DESC")
        for row in cursor.fetchall():
            self._cache[row[0].lower()] = row[1]

    # ------------------------------------------------------------------
    # Core lookup
    # ------------------------------------------------------------------

    def resolve(self, app_name: str) -> Optional[str]:
        """
        Resolve an app name to its executable path.
        Priority: user-taught > discovered > known aliases > system defaults.
        Returns None if not found.
        """
        if not app_name:
            return None

        name_lower = app_name.lower().strip()

        # 1. Direct cache hit (already discovered/taught)
        if name_lower in self._cache:
            self._record_usage(name_lower)
            return self._cache[name_lower]

        # 2. Try alias matching
        for alias, exe_patterns in KNOWN_ALIASES.items():
            if alias in name_lower or name_lower in alias:
                # Try to find the actual exe on disk
                found_path = self._find_exe_on_disk(exe_patterns)
                if found_path:
                    self._store_discovered(name_lower, found_path, "alias")
                    return found_path
                # Return pattern as fallback (Windows will resolve from PATH)
                if exe_patterns:
                    path = exe_patterns[0]
                    self._store_discovered(name_lower, path, "alias_fallback")
                    return path

        # 3. Try Windows Registry
        path = self._scan_registry(name_lower)
        if path:
            self._store_discovered(name_lower, path, "registry")
            return path

        # 4. Try filesystem scan
        path = self._scan_filesystem(name_lower)
        if path:
            self._store_discovered(name_lower, path, "filesystem")
            return path

        logger.warning(f"[AppRegistry] Could not resolve app: '{app_name}'")
        return None

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def teach(self, app_name: str, path: str):
        """
        User teaches AERIS a new app path.
        User-taught mappings have highest priority.
        """
        name_lower = app_name.lower().strip()
        self.conn.execute("""
            INSERT INTO apps (name, path, source, confidence, user_taught)
            VALUES (?, ?, 'user', 1.0, 1)
            ON CONFLICT(name) DO UPDATE SET
                path = ?,
                source = 'user',
                user_taught = 1,
                confidence = 1.0
        """, (name_lower, path, path))
        self.conn.commit()
        self._cache[name_lower] = path
        logger.info(f"[AppRegistry] User taught: '{app_name}' → {path}")

    def learn_user_preference(self, category: str, app_name: str):
        """
        Learn that for a category (e.g., 'browser'), the user prefers this app.
        """
        path = self.resolve(app_name)
        if path:
            self.teach(category, path)
            self.teach(app_name, path)
            logger.info(f"[AppRegistry] Learned preference: {category} → {app_name} ({path})")
            return True
        return False

    # ------------------------------------------------------------------
    # Discovery helpers
    # ------------------------------------------------------------------

    def _scan_registry(self, app_name: str) -> Optional[str]:
        """Scan Windows registry for app installations."""
        if sys.platform != "win32":
            return None

        try:
            import winreg

            reg_paths = [
                r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths",
                r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\App Paths",
            ]

            for hive in [winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER]:
                for reg_path in reg_paths:
                    try:
                        key = winreg.OpenKey(hive, reg_path)
                        i = 0
                        while True:
                            try:
                                subkey_name = winreg.EnumKey(key, i)
                                if app_name in subkey_name.lower():
                                    subkey = winreg.OpenKey(key, subkey_name)
                                    try:
                                        val, _ = winreg.QueryValueEx(subkey, "")
                                        if val and os.path.exists(val):
                                            return val
                                    except Exception:
                                        pass
                                    finally:
                                        winreg.CloseKey(subkey)
                                i += 1
                            except OSError:
                                break
                        winreg.CloseKey(key)
                    except Exception:
                        continue
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"[AppRegistry] Registry scan error: {e}")

        return None

    def _scan_filesystem(self, app_name: str) -> Optional[str]:
        """Scan common installation directories for matching executables."""
        exe_name = app_name.replace(" ", "") + ".exe"
        search_names = [
            exe_name,
            app_name.replace(" ", "_") + ".exe",
            app_name + ".exe",
        ]

        for base_path in SCAN_PATHS:
            if not os.path.exists(base_path):
                continue
            try:
                for root, dirs, files in os.walk(base_path):
                    # Limit depth to avoid too-deep scanning
                    depth = root.replace(base_path, "").count(os.sep)
                    if depth > 3:
                        del dirs[:]
                        continue

                    for fname in files:
                        fname_lower = fname.lower()
                        for search in search_names:
                            if fname_lower == search.lower():
                                return os.path.join(root, fname)
            except PermissionError:
                continue
            except Exception as e:
                logger.debug(f"[AppRegistry] Scan error in {base_path}: {e}")

        return None

    def _find_exe_on_disk(self, exe_patterns: List[str]) -> Optional[str]:
        """Look for any of the given exe names in known locations."""
        for exe in exe_patterns:
            if not exe.endswith(".exe"):
                continue
            # Check PATH
            import shutil
            found = shutil.which(exe)
            if found:
                return found
            # Check Program Files
            for base in [r"C:\Program Files", r"C:\Program Files (x86)"]:
                for root, _, files in os.walk(base):
                    depth = root.replace(base, "").count(os.sep)
                    if depth > 3:
                        break
                    if exe.lower() in [f.lower() for f in files]:
                        return os.path.join(root, exe)
        return None

    def _store_discovered(self, name: str, path: str, source: str):
        """Store a discovered app mapping."""
        try:
            self.conn.execute("""
                INSERT INTO apps (name, path, source, confidence)
                VALUES (?, ?, ?, 0.8)
                ON CONFLICT(name) DO NOTHING
            """, (name, path, source))
            self.conn.commit()
            self._cache[name] = path
        except Exception as e:
            logger.debug(f"[AppRegistry] Store error: {e}")

    def _record_usage(self, name: str):
        """Update usage statistics."""
        try:
            self.conn.execute("""
                UPDATE apps SET use_count = use_count + 1, last_used = ?
                WHERE name = ?
            """, (time.time(), name))
            self.conn.commit()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def list_known_apps(self) -> List[dict]:
        """Return all known app mappings."""
        cursor = self.conn.execute("""
            SELECT name, path, source, use_count, user_taught
            FROM apps ORDER BY use_count DESC, user_taught DESC
        """)
        return [
            {"name": r[0], "path": r[1], "source": r[2], "uses": r[3], "user_taught": bool(r[4])}
            for r in cursor.fetchall()
        ]

    def needs_clarification(self, app_name: str) -> bool:
        """Return True if we could not resolve the app and need to ask the user."""
        return self.resolve(app_name) is None
