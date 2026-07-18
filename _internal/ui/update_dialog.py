"""
ui/update_dialog.py
Beautiful update notification + in-app download/apply system for Nova Ultimate Suite.
No separate updater.exe required — updates run directly inside Nova_Ultimate_Suite.exe.
"""
import sys
import json
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QGraphicsDropShadowEffect,
    QScrollArea, QWidget, QApplication, QProgressBar,
)
from PyQt6.QtCore import (
    Qt, QPropertyAnimation, QEasingCurve, QTimer,
    QThread, pyqtSignal,
)
from PyQt6.QtGui import QFont, QColor, QLinearGradient, QPainter, QPainterPath


# ─── Skipped-version persistence ────────────────────────────────────────────
def _skipped_version_path() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent / "settings" / "skipped_version.txt"
    return Path(__file__).resolve().parent.parent / "settings" / "skipped_version.txt"


def get_skipped_version() -> str:
    p = _skipped_version_path()
    try:
        return p.read_text(encoding="utf-8").strip() if p.exists() else ""
    except Exception:
        return ""


def set_skipped_version(version: str) -> None:
    p = _skipped_version_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(version, encoding="utf-8")
    except Exception:
        pass


# ─── Background download + apply worker ─────────────────────────────────────
class InAppUpdateWorker(QThread):
    progress   = pyqtSignal(int, str)   # (percent, status_message)
    finished   = pyqtSignal(bool, str)  # (success, message)

    GITHUB_RAW = "https://raw.githubusercontent.com/phearun01/nova-suite-updates/main"

    def __init__(self, download_url: str, new_version: str):
        super().__init__()
        self.download_url = download_url
        self.new_version  = new_version

    # Resolve app root (works for both frozen EXE and dev mode)
    @property
    def app_dir(self) -> Path:
        if getattr(sys, "frozen", False):
            return Path(sys.executable).parent
        return Path(__file__).resolve().parent.parent

    def run(self):
        try:
            app_dir   = self.app_dir
            tmp_dir   = app_dir / "update_temp"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            zip_path  = tmp_dir / "patch.zip"

            # ── Step 1: Download ────────────────────────────────────────────
            self.progress.emit(5, "Connecting to server...")
            req = urllib.request.Request(
                self.download_url,
                headers={"User-Agent": "Mozilla/5.0 Nova-Updater/2.3.8"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                chunk = 32768
                with open(zip_path, "wb") as f:
                    while True:
                        buf = resp.read(chunk)
                        if not buf:
                            break
                        f.write(buf)
                        downloaded += len(buf)
                        if total:
                            pct = int(downloaded / total * 60)
                            kb = downloaded // 1024
                            self.progress.emit(5 + pct, f"Downloading... {kb} KB")

            self.progress.emit(65, "Download complete. Verifying...")

            # ── Step 2: Verify ZIP ──────────────────────────────────────────
            if not zipfile.is_zipfile(zip_path):
                self.finished.emit(False, "Downloaded file is not a valid ZIP.")
                return

            # ── Step 3: Backup changed files ────────────────────────────────
            self.progress.emit(70, "Backing up current files...")
            backup_dir = tmp_dir / "backup"
            backup_dir.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(zip_path, "r") as zf:
                members = [m for m in zf.namelist() if not m.endswith("/")]
                self.progress.emit(72, f"Applying {len(members)} file(s)...")

                # ── Step 4: Extract & apply ────────────────────────────────
                for i, member in enumerate(members):
                    # Strip leading folder (Nova_Ultimate_Suite/) from path
                    parts = Path(member).parts
                    if len(parts) <= 1:
                        continue
                    rel_path = Path(*parts[1:])   # e.g. _internal/ui/video_dubber.py
                    dest = app_dir / rel_path

                    # Backup existing file
                    if dest.exists():
                        bk = backup_dir / rel_path
                        bk.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(dest, bk)

                    # Write new file
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    data = zf.read(member)
                    dest.write_bytes(data)

                    pct = 72 + int((i + 1) / len(members) * 20)
                    self.progress.emit(pct, f"Updated: {rel_path.name}")

            # ── Step 5: Write new version.json to app dir ─────────────────
            self.progress.emit(93, "Updating version info...")
            try:
                vj_path = app_dir / "version.json"
                vj_data: dict = {}
                if vj_path.exists():
                    try:
                        vj_data = json.loads(vj_path.read_text(encoding="utf-8"))
                    except Exception:
                        pass
                vj_data["version"]           = self.new_version
                vj_data["installed_version"] = self.new_version
                vj_path.write_text(json.dumps(vj_data, indent=2, ensure_ascii=False), "utf-8")
            except Exception as e:
                pass  # Non-fatal

            # ── Step 6: Cleanup temp ────────────────────────────────────────
            self.progress.emit(97, "Cleaning up...")
            try:
                shutil.rmtree(tmp_dir)
            except Exception:
                pass

            self.progress.emit(100, "Update complete!")
            self.finished.emit(True, "")

        except Exception as e:
            self.finished.emit(False, str(e))


# ─── Progress dialog shown during download / apply ──────────────────────────
class UpdateProgressDialog(QDialog):
    def __init__(self, worker: InAppUpdateWorker, new_version: str, parent=None):
        super().__init__(parent)
        self.worker      = worker
        self.new_version = new_version
        self._success    = False

        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setModal(True)
        self.setFixedSize(420, 220)
        self._build_ui()

        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)

        card = QFrame()
        card.setObjectName("pcard")
        card.setStyleSheet("""
            QFrame#pcard {
                background: #0f1117;
                border: 1px solid #2a2d3e;
                border-radius: 16px;
            }
        """)
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(35)
        shadow.setColor(QColor(0, 0, 0, 160))
        shadow.setOffset(0, 6)
        card.setGraphicsEffect(shadow)

        lay = QVBoxLayout(card)
        lay.setContentsMargins(28, 24, 28, 24)
        lay.setSpacing(14)

        # Title
        title = QLabel(f"⬇️  Updating to v{self.new_version}")
        title.setStyleSheet("color: #f1f5f9; font-size: 16px; font-weight: bold;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(title)

        # Status label
        self.lbl_status = QLabel("Preparing...")
        self.lbl_status.setStyleSheet("color: #94a3b8; font-size: 12px;")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.lbl_status)

        # Progress bar
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setFixedHeight(10)
        self.bar.setTextVisible(False)
        self.bar.setStyleSheet("""
            QProgressBar {
                background: #1e2030;
                border: none;
                border-radius: 5px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #6366f1, stop:1 #10b981);
                border-radius: 5px;
            }
        """)
        lay.addWidget(self.bar)

        # Percent label
        self.lbl_pct = QLabel("0%")
        self.lbl_pct.setStyleSheet("color: #64748b; font-size: 11px;")
        self.lbl_pct.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.lbl_pct)

        # Note
        note = QLabel("Please keep the app open. It will restart automatically.")
        note.setStyleSheet("color: #475569; font-size: 10px;")
        note.setWordWrap(True)
        note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(note)

        root.addWidget(card)

    def _on_progress(self, pct: int, msg: str):
        self.bar.setValue(pct)
        self.lbl_pct.setText(f"{pct}%")
        self.lbl_status.setText(msg)

    def _on_finished(self, success: bool, error: str):
        self._success = success
        if success:
            self.lbl_status.setText("✅  Update applied! Restarting...")
            self.bar.setValue(100)
            QTimer.singleShot(1500, self._restart_app)
        else:
            self.lbl_status.setText(f"❌  {error}")
            QTimer.singleShot(3000, self.reject)

    def _restart_app(self):
        """Relaunch Nova_Ultimate_Suite.exe and quit current instance."""
        if getattr(sys, "frozen", False):
            exe = Path(sys.executable)
        else:
            exe = Path(__file__).resolve().parent.parent / "desktop_app.py"

        try:
            if getattr(sys, "frozen", False):
                subprocess.Popen([str(exe)])
            else:
                subprocess.Popen([sys.executable, str(exe)])
        except Exception:
            pass

        self.accept()
        QTimer.singleShot(300, QApplication.quit)


# ─── Main Update Info Dialog ──────────────────────────────────────────────────
class UpdateDialog(QDialog):
    """
    Show version info + Skip / Update Now buttons.
    On Update Now → InAppUpdateWorker handles download+apply inside the EXE.
    """
    def __init__(self, parent=None, latest_version="", download_url="", notes=""):
        super().__init__(parent)
        self.latest_version = latest_version
        self.download_url   = download_url
        self.notes          = notes

        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setModal(True)
        self.setFixedSize(460, 420)

        self._build_ui()
        self._apply_entrance_animation()

    # ── UI ───────────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)

        card = QFrame()
        card.setObjectName("card")
        card.setStyleSheet("""
            QFrame#card {
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #0f1117, stop:1 #13141f);
                border: 1px solid #2a2d3e;
                border-radius: 18px;
            }
        """)
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(40)
        shadow.setColor(QColor(0, 0, 0, 180))
        shadow.setOffset(0, 8)
        card.setGraphicsEffect(shadow)

        lay = QVBoxLayout(card)
        lay.setContentsMargins(32, 28, 32, 28)
        lay.setSpacing(0)

        # Close button
        close_row = QHBoxLayout()
        close_row.addStretch()
        btn_x = QPushButton("✕")
        btn_x.setFixedSize(28, 28)
        btn_x.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_x.setStyleSheet("""
            QPushButton { background:#1e2030; color:#64748b; border:none;
                          border-radius:14px; font-size:11px; font-weight:bold; }
            QPushButton:hover { background:#ef4444; color:white; }
        """)
        btn_x.clicked.connect(self.reject)
        close_row.addWidget(btn_x)
        lay.addLayout(close_row)

        # Logo
        logo = QLabel("✨")
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo.setStyleSheet("font-size: 48px; margin-bottom:2px;")
        lay.addWidget(logo)

        # App name
        name = QLabel("Nova Ultimate Suite")
        name.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name.setStyleSheet("color:#f1f5f9; margin-bottom:4px;")
        lay.addWidget(name)

        # Version badge
        br = QHBoxLayout()
        br.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge = QLabel(f"  🆕  Version {self.latest_version} is available  ")
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #312e81, stop:1 #1e3a5f);
                color:#818cf8; border:1px solid #4f46e5;
                border-radius:20px; font-size:13px; font-weight:bold;
                padding:6px 18px; margin-top:4px;
            }
        """)
        br.addWidget(badge)
        lay.addLayout(br)

        # Divider
        div = QFrame()
        div.setFixedHeight(1)
        div.setStyleSheet("background:#1e2030; margin:16px 0 10px 0;")
        lay.addWidget(div)

        # Notes header
        nh = QLabel("📋  What's New")
        nh.setStyleSheet("color:#94a3b8; font-size:11px; font-weight:600; letter-spacing:1px; margin-bottom:6px;")
        lay.addWidget(nh)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(100)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("""
            QScrollArea { background:transparent; border:none; }
            QScrollBar:vertical { background:#0f1117; width:4px; border-radius:2px; }
            QScrollBar::handle:vertical { background:#3b82f6; border-radius:2px; min-height:20px; }
        """)
        nw = QWidget()
        nw.setStyleSheet("background:transparent;")
        nl = QVBoxLayout(nw)
        nl.setContentsMargins(0, 0, 8, 0)
        nl.setSpacing(4)
        raw = self.notes or "• Bug fixes and performance improvements"
        for line in raw.split("\n"):
            line = line.strip()
            if not line:
                continue
            if not (line.startswith("•") or line.startswith("-")):
                line = f"• {line}"
            lbl = QLabel(line)
            lbl.setWordWrap(True)
            lbl.setStyleSheet("color:#cbd5e1; font-size:12px; padding:1px 0;")
            nl.addWidget(lbl)
        nl.addStretch()
        scroll.setWidget(nw)
        lay.addWidget(scroll)

        lay.addSpacing(16)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)

        self.btn_skip = QPushButton("⏭  SKIP THIS VERSION")
        self.btn_skip.setFixedHeight(44)
        self.btn_skip.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_skip.setStyleSheet("""
            QPushButton {
                background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #b45309,stop:1 #d97706);
                color:#fff; border:none; border-radius:22px;
                font-size:12px; font-weight:bold;
            }
            QPushButton:hover { background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 #d97706,stop:1 #f59e0b); }
        """)
        self.btn_skip.clicked.connect(self._on_skip)
        btn_row.addWidget(self.btn_skip)

        self.btn_update = QPushButton("▶  UPDATE NOW")
        self.btn_update.setFixedHeight(44)
        self.btn_update.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_update.setStyleSheet("""
            QPushButton {
                background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #059669,stop:1 #10b981);
                color:#fff; border:none; border-radius:22px;
                font-size:12px; font-weight:bold;
            }
            QPushButton:hover { background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 #10b981,stop:1 #34d399); }
        """)
        self.btn_update.clicked.connect(self._on_update_now)
        btn_row.addWidget(self.btn_update)

        lay.addLayout(btn_row)
        root.addWidget(card)

    # ── Entrance animation ───────────────────────────────────────────────────
    def _apply_entrance_animation(self):
        self.setWindowOpacity(0.0)
        self._fade = QPropertyAnimation(self, b"windowOpacity")
        self._fade.setDuration(280)
        self._fade.setStartValue(0.0)
        self._fade.setEndValue(1.0)
        self._fade.setEasingCurve(QEasingCurve.Type.OutCubic)
        QTimer.singleShot(50, self._fade.start)

    # ── Actions ──────────────────────────────────────────────────────────────
    def _on_skip(self):
        set_skipped_version(self.latest_version)
        self.reject()

    def _on_update_now(self):
        """Start in-app download + apply inside Nova_Ultimate_Suite.exe."""
        self.btn_update.setEnabled(False)
        self.btn_skip.setEnabled(False)
        self.btn_update.setText("⏳  Preparing...")

        worker = InAppUpdateWorker(self.download_url, self.latest_version)
        prog   = UpdateProgressDialog(worker, self.latest_version, self)
        self.accept()   # close info dialog
        prog.exec()     # show progress dialog (blocks until done / restart)

    # ── Dragging ─────────────────────────────────────────────────────────────
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton and hasattr(self, "_drag_pos"):
            self.move(event.globalPosition().toPoint() - self._drag_pos)
