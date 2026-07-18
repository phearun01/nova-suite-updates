import urllib.request
import json
import logging
from PyQt6.QtCore import QThread, pyqtSignal
from version import __version__ as CURRENT_VERSION

# Remote update endpoint — must serve JSON: {"version": "X.Y.Z", "url": "...", "notes": "..."}
UPDATE_URL = "https://raw.githubusercontent.com/phearun01/nova-suite-updates/main/version.json"

class UpdateCheckerThread(QThread):
    # Signals: (has_update, latest_version, download_url, release_notes)
    finished = pyqtSignal(bool, str, str, str)

    def run(self):
        try:
            req = urllib.request.Request(
                UPDATE_URL, 
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode('utf-8'))
                
            latest_version = data.get("version", "1.0.0")
            download_url = data.get("url", "")
            release_notes = data.get("notes", "No release notes available.")
            
            # Simple semantic version comparison
            v_curr = list(map(int, CURRENT_VERSION.split(".")))
            v_late = list(map(int, latest_version.split(".")))
            
            has_update = v_late > v_curr
            self.finished.emit(has_update, latest_version, download_url, release_notes)
            
        except Exception as e:
            logging.warning(f"Update check failed: {e}")
            # Fallback to no update on errors
            self.finished.emit(False, CURRENT_VERSION, "", "")
