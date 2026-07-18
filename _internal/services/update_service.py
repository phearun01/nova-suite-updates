import urllib.request
import json
import logging
import sys
from pathlib import Path
from PyQt6.QtCore import QThread, pyqtSignal

# Remote update endpoint — must serve JSON: {"version": "X.Y.Z", "url": "...", "notes": "..."}
UPDATE_URL = "https://raw.githubusercontent.com/phearun01/nova-suite-updates/main/version.json"


def _get_installed_version() -> str:
    """
    Read the current installed version.
    Priority:
      1. version.json in the app directory (written by updater after each patch)
      2. version.py module (source / first-run fallback)
    """
    # In frozen (compiled) app — check app dir for version.json first
    if getattr(sys, "frozen", False):
        app_dir = Path(sys.executable).parent
        vj_path = app_dir / "version.json"
        if vj_path.exists():
            try:
                data = json.loads(vj_path.read_text(encoding="utf-8"))
                # Updater writes the remote version.json which has "version" key
                v = data.get("installed_version") or data.get("version")
                if v and v != "0.0.0":
                    return v
            except Exception:
                pass

    # Fallback: read from version.py module
    try:
        from version import __version__
        return __version__
    except Exception:
        return "1.0.0"


CURRENT_VERSION: str = _get_installed_version()


class UpdateCheckerThread(QThread):
    # Signals: (has_update, latest_version, download_url, release_notes)
    finished = pyqtSignal(bool, str, str, str)

    def run(self):
        try:
            # Re-read installed version each time (in case updater ran)
            current = _get_installed_version()

            req = urllib.request.Request(
                UPDATE_URL,
                headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode("utf-8"))

            latest_version = data.get("version", "1.0.0")
            download_url   = data.get("url", "")
            release_notes  = data.get("notes", "No release notes available.")

            # Semantic version comparison
            v_curr = list(map(int, current.split(".")))
            v_late = list(map(int, latest_version.split(".")))

            has_update = v_late > v_curr
            self.finished.emit(has_update, latest_version, download_url, release_notes)

        except Exception as e:
            logging.warning(f"Update check failed: {e}")
            self.finished.emit(False, CURRENT_VERSION, "", "")
