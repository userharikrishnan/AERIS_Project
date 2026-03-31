"""
AERIS Update Manager
Local file-based model update system.

Architecture:
- Global model updates come as .zip packages dropped into the `updates/` folder
  (or downloaded manually from a release URL in the future)
- Personal adaptation data (learned preferences, entity overrides) is ALWAYS preserved
- Update process: unzip → validate → backup old models → apply → reload
- Versioning tracked in `config/update_manifest.json`

Why local file-based?
- No cloud infra needed at this stage
- User controls when updates are applied
- Safe, auditable, reversible
- Future: can add auto-download from a release URL with one small addition
"""

import os
import json
import shutil
import zipfile
import hashlib
import logging
import time
from typing import Optional, Dict

logger = logging.getLogger(__name__)

MANIFEST_PATH = "config/update_manifest.json"
UPDATES_DIR = "updates"
CHECKPOINTS_DIR = "checkpoints"
BACKUP_DIR = "checkpoints_backup"

DEFAULT_MANIFEST = {
    "aeris_version": "0.1.0",
    "global_model_version": "0.1.0",
    "personal_model_version": "0.1.0",
    "last_updated": None,
    "update_history": [],
    "checksums": {}
}


class UpdateManager:
    """
    Manages AERIS model and system updates.
    Handles discovery, validation, backup, and application of update packages.
    """

    def __init__(self):
        os.makedirs(UPDATES_DIR, exist_ok=True)
        os.makedirs("config", exist_ok=True)
        self.manifest = self._load_manifest()

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    def _load_manifest(self) -> dict:
        if os.path.exists(MANIFEST_PATH):
            try:
                with open(MANIFEST_PATH, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        manifest = dict(DEFAULT_MANIFEST)
        self._save_manifest(manifest)
        return manifest

    def _save_manifest(self, manifest: dict = None):
        m = manifest or self.manifest
        with open(MANIFEST_PATH, "w") as f:
            json.dump(m, f, indent=2)

    # ------------------------------------------------------------------
    # Update discovery
    # ------------------------------------------------------------------

    def check_for_updates(self) -> dict:
        """
        Scan the updates/ directory for pending update packages.
        Returns info about available updates.
        """
        pending = []

        if not os.path.exists(UPDATES_DIR):
            return {"pending": [], "message": "No updates directory found."}

        for fname in os.listdir(UPDATES_DIR):
            if fname.endswith(".zip"):
                fpath = os.path.join(UPDATES_DIR, fname)
                size_mb = os.path.getsize(fpath) / (1024 * 1024)
                pending.append({
                    "file": fname,
                    "path": fpath,
                    "size_mb": round(size_mb, 2),
                    "modified": os.path.getmtime(fpath)
                })

        return {
            "pending": pending,
            "count": len(pending),
            "current_version": self.manifest.get("global_model_version"),
            "last_updated": self.manifest.get("last_updated")
        }

    # ------------------------------------------------------------------
    # Update application
    # ------------------------------------------------------------------

    def apply_update(self, update_file: str) -> dict:
        """
        Apply a model update package.

        Package structure (zip):
        - manifest.json  (version info, required models)
        - aeris_slm.pt (optional)
        - aeris_slm_best.pt (optional)
        - intent_classifier.pt (optional)
        - plan_scorer.pt (optional)
        - tokenizer.json (optional)
        - CHANGELOG.md (optional)
        """
        if not os.path.exists(update_file):
            return {"success": False, "error": f"Update file not found: {update_file}"}

        logger.info(f"[UpdateManager] Applying update: {update_file}")

        try:
            # Step 1: Validate the package
            validation = self._validate_package(update_file)
            if not validation["valid"]:
                return {"success": False, "error": validation["reason"]}

            # Step 2: Backup current models
            backup_path = self._backup_models()
            logger.info(f"[UpdateManager] Models backed up to: {backup_path}")

            # Step 3: Extract and apply
            with zipfile.ZipFile(update_file, "r") as zf:
                # Read package manifest
                pkg_manifest = {}
                if "manifest.json" in zf.namelist():
                    pkg_manifest = json.loads(zf.read("manifest.json").decode("utf-8"))

                new_version = pkg_manifest.get("version", "unknown")
                files_updated = []

                # Apply model files
                model_files = [f for f in zf.namelist() if f.endswith((".pt", ".json", ".md"))]
                for fname in model_files:
                    if fname == "manifest.json":
                        continue
                    dest = os.path.join(CHECKPOINTS_DIR, fname)
                    with zf.open(fname) as src, open(dest, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    files_updated.append(fname)
                    logger.info(f"[UpdateManager] Updated: {fname}")

            # Step 4: Update manifest
            self.manifest["global_model_version"] = new_version
            self.manifest["last_updated"] = time.time()
            self.manifest["update_history"].append({
                "file": os.path.basename(update_file),
                "version": new_version,
                "applied_at": time.time(),
                "files_updated": files_updated
            })
            self._save_manifest()

            # Step 5: Move processed update to applied/
            applied_dir = os.path.join(UPDATES_DIR, "applied")
            os.makedirs(applied_dir, exist_ok=True)
            shutil.move(update_file, os.path.join(applied_dir, os.path.basename(update_file)))

            return {
                "success": True,
                "version_applied": new_version,
                "files_updated": files_updated,
                "backup_path": backup_path,
                "message": f"Successfully updated to version {new_version}. Restart AERIS to load new models."
            }

        except Exception as e:
            logger.error(f"[UpdateManager] Update failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def _validate_package(self, zip_path: str) -> dict:
        """Basic validation of update package."""
        try:
            if not zipfile.is_zipfile(zip_path):
                return {"valid": False, "reason": "Not a valid zip file"}

            with zipfile.ZipFile(zip_path, "r") as zf:
                names = zf.namelist()
                # Must have at least one model file or a changelog
                has_content = any(f.endswith((".pt", ".json")) for f in names)
                if not has_content:
                    return {"valid": False, "reason": "Package contains no model files"}

            return {"valid": True}
        except Exception as e:
            return {"valid": False, "reason": str(e)}

    def _backup_models(self) -> str:
        """Backup current checkpoints before applying update."""
        ts = time.strftime("%Y%m%d_%H%M%S")
        backup_path = f"{BACKUP_DIR}_{ts}"

        if os.path.exists(CHECKPOINTS_DIR):
            shutil.copytree(CHECKPOINTS_DIR, backup_path)

        return backup_path

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------

    def rollback_to_backup(self, backup_path: str) -> dict:
        """Restore models from a backup."""
        if not os.path.exists(backup_path):
            return {"success": False, "error": f"Backup not found: {backup_path}"}

        try:
            if os.path.exists(CHECKPOINTS_DIR):
                shutil.rmtree(CHECKPOINTS_DIR)
            shutil.copytree(backup_path, CHECKPOINTS_DIR)
            logger.info(f"[UpdateManager] Rolled back to backup: {backup_path}")
            return {"success": True, "restored_from": backup_path}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------
    # Version info
    # ------------------------------------------------------------------

    def get_version_info(self) -> dict:
        return {
            "aeris_version": self.manifest.get("aeris_version"),
            "global_model_version": self.manifest.get("global_model_version"),
            "personal_model_version": self.manifest.get("personal_model_version"),
            "last_updated": self.manifest.get("last_updated"),
            "update_count": len(self.manifest.get("update_history", []))
        }
