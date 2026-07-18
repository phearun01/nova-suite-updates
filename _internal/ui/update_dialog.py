"""
ui/update_dialog.py
Beautiful update notification dialog for Nova Ultimate Suite.
Shows version, release notes, and Skip / Update Now actions.
"""
import sys
import json
import subprocess
from pathlib import Path

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QGraphicsDropShadowEffect,
    QScrollArea, QWidget, QApplication,
)
from PyQt6.QtCore import Qt, QPropertyAnimation, QEasingCurve, QRect, QTimer
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


# ─── Animated gradient title label ──────────────────────────────────────────
class GradientLabel(QLabel):
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        gradient = QLinearGradient(0, 0, self.width(), 0)
        gradient.setColorAt(0.0, QColor("#6366f1"))
        gradient.setColorAt(0.5, QColor("#8b5cf6"))
        gradient.setColorAt(1.0, QColor("#06b6d4"))
        painter.setPen(Qt.PenStyle.NoPen)
        path = QPainterPath()
        path.addText(
            (self.width() - self.fontMetrics().horizontalAdvance(self.text())) / 2,
            self.fontMetrics().ascent() + 4,
            self.font(),
            self.text(),
        )
        painter.setBrush(gradient)
        painter.drawPath(path)


# ─── Main Update Dialog ──────────────────────────────────────────────────────
class UpdateDialog(QDialog):
    """
    Beautiful update notification dialog.

    Usage:
        dlg = UpdateDialog(parent, latest_version="2.3.8",
                           download_url="...", notes="• Bug fix\n• Performance")
        dlg.exec()
    """

    def __init__(
        self,
        parent=None,
        latest_version: str = "",
        download_url: str = "",
        notes: str = "",
    ):
        super().__init__(parent)
        self.latest_version = latest_version
        self.download_url = download_url
        self.notes = notes

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

    # ── UI construction ──────────────────────────────────────────────────────
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)

        # Card frame
        card = QFrame(self)
        card.setObjectName("card")
        card.setStyleSheet("""
            QFrame#card {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
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

        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(32, 28, 32, 28)
        card_layout.setSpacing(0)

        # ── Close button ─────────────────────────────────────────────────────
        close_row = QHBoxLayout()
        close_row.addStretch()
        btn_close = QPushButton("✕")
        btn_close.setFixedSize(28, 28)
        btn_close.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_close.setStyleSheet("""
            QPushButton {
                background: #1e2030;
                color: #64748b;
                border: none;
                border-radius: 14px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover { background: #ef4444; color: white; }
        """)
        btn_close.clicked.connect(self.reject)
        close_row.addWidget(btn_close)
        card_layout.addLayout(close_row)

        # ── Logo / icon ───────────────────────────────────────────────────────
        logo_lbl = QLabel("✨")
        logo_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo_lbl.setStyleSheet("font-size: 48px; margin-bottom: 2px;")
        card_layout.addWidget(logo_lbl)

        # ── App name ──────────────────────────────────────────────────────────
        name_lbl = QLabel("Nova Ultimate Suite")
        name_font = QFont("Segoe UI", 18, QFont.Weight.Bold)
        name_lbl.setFont(name_font)
        name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name_lbl.setStyleSheet("color: #f1f5f9; margin-bottom: 4px;")
        card_layout.addWidget(name_lbl)

        # ── Version badge ─────────────────────────────────────────────────────
        badge_row = QHBoxLayout()
        badge_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        version_badge = QLabel(f"  🆕  Version {self.latest_version} is available  ")
        version_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        version_badge.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #312e81, stop:1 #1e3a5f);
                color: #818cf8;
                border: 1px solid #4f46e5;
                border-radius: 20px;
                font-size: 13px;
                font-weight: bold;
                padding: 6px 18px;
                margin-top: 4px;
            }
        """)
        badge_row.addWidget(version_badge)
        card_layout.addLayout(badge_row)

        # ── Divider ───────────────────────────────────────────────────────────
        divider = QFrame()
        divider.setFixedHeight(1)
        divider.setStyleSheet("background: #1e2030; margin: 16px 0 10px 0;")
        card_layout.addWidget(divider)

        # ── Release notes ─────────────────────────────────────────────────────
        notes_header = QLabel("📋  What's New")
        notes_header.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: 600; letter-spacing: 1px; margin-bottom: 6px;")
        card_layout.addWidget(notes_header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(100)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("""
            QScrollArea { background: transparent; border: none; }
            QScrollBar:vertical {
                background: #0f1117; width: 4px; border-radius: 2px;
            }
            QScrollBar::handle:vertical {
                background: #3b82f6; border-radius: 2px; min-height: 20px;
            }
        """)

        notes_widget = QWidget()
        notes_widget.setStyleSheet("background: transparent;")
        notes_layout = QVBoxLayout(notes_widget)
        notes_layout.setContentsMargins(0, 0, 8, 0)
        notes_layout.setSpacing(4)

        # Parse notes into bullet lines
        raw_notes = self.notes or "• Bug fixes and performance improvements"
        for line in raw_notes.split("\n"):
            line = line.strip()
            if not line:
                continue
            if not line.startswith("•") and not line.startswith("-"):
                line = f"• {line}"
            note_lbl = QLabel(line)
            note_lbl.setWordWrap(True)
            note_lbl.setStyleSheet("color: #cbd5e1; font-size: 12px; padding: 1px 0;")
            notes_layout.addWidget(note_lbl)
        notes_layout.addStretch()

        scroll.setWidget(notes_widget)
        card_layout.addWidget(scroll)

        card_layout.addSpacing(16)

        # ── Buttons ───────────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)

        # Skip button
        self.btn_skip = QPushButton("⏭  SKIP THIS VERSION")
        self.btn_skip.setFixedHeight(44)
        self.btn_skip.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_skip.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #b45309, stop:1 #d97706);
                color: #fff;
                border: none;
                border-radius: 22px;
                font-size: 12px;
                font-weight: bold;
                letter-spacing: 0.5px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #d97706, stop:1 #f59e0b);
            }
            QPushButton:pressed { opacity: 0.85; }
        """)
        self.btn_skip.clicked.connect(self._on_skip)
        btn_row.addWidget(self.btn_skip)

        # Update now button
        self.btn_update = QPushButton("▶  UPDATE NOW")
        self.btn_update.setFixedHeight(44)
        self.btn_update.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_update.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #059669, stop:1 #10b981);
                color: #fff;
                border: none;
                border-radius: 22px;
                font-size: 12px;
                font-weight: bold;
                letter-spacing: 0.5px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #10b981, stop:1 #34d399);
            }
            QPushButton:pressed { opacity: 0.85; }
        """)
        self.btn_update.clicked.connect(self._on_update_now)
        btn_row.addWidget(self.btn_update)

        card_layout.addLayout(btn_row)
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

    # ── Button actions ───────────────────────────────────────────────────────
    def _on_skip(self):
        """Remember this version so dialog doesn't show again for it."""
        set_skipped_version(self.latest_version)
        self.reject()

    def _on_update_now(self):
        """Launch updater.exe with the correct GitHub server URL."""
        self.btn_update.setEnabled(False)
        self.btn_update.setText("⏳  Launching updater...")

        # GitHub raw URL (without /version.json) — updater detects GitHub mode from this
        GITHUB_SERVER_URL = "https://raw.githubusercontent.com/phearun01/nova-suite-updates/main"

        if getattr(sys, "frozen", False):
            exe_dir      = Path(sys.executable).parent
            updater_path = exe_dir / "updater.exe"
            app_dir      = exe_dir
        else:
            app_dir      = Path(__file__).resolve().parent.parent
            updater_path = app_dir / "scripts" / "build" / "updater.py"

        if updater_path.exists():
            cmd = (
                [str(updater_path), "--app-dir", str(app_dir), "--url", GITHUB_SERVER_URL]
                if getattr(sys, "frozen", False)
                else [sys.executable, str(updater_path), "--app-dir", str(app_dir), "--url", GITHUB_SERVER_URL]
            )
            subprocess.Popen(cmd)
            self.accept()
            QTimer.singleShot(500, QApplication.quit)
        else:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, "Updater Not Found",
                f"updater.exe not found at:\n{updater_path}\n\nPlease update manually.",
            )
            self.btn_update.setEnabled(True)
            self.btn_update.setText("▶  UPDATE NOW")


    # ── Allow dragging the frameless dialog ──────────────────────────────────
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton and hasattr(self, "_drag_pos"):
            self.move(event.globalPosition().toPoint() - self._drag_pos)
