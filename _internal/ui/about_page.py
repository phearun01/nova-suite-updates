from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QPushButton, QMessageBox,
    QScrollArea
)
from PyQt6.QtCore import Qt
import webbrowser

# Import Update Service
from services.update_service import UpdateCheckerThread, CURRENT_VERSION
from ui.update_dialog import UpdateDialog, get_skipped_version

class AboutPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.update_thread = None
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { background-color: transparent; }")
        
        scroll_content = QWidget()
        scroll_content.setStyleSheet("background-color: transparent;")
        layout = QVBoxLayout(scroll_content)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(25)
        
        card = QFrame()
        card.setFixedWidth(500)
        card.setStyleSheet("""
            QFrame {
                background-color: #13141f; 
                border: 1px solid #1e2030; 
                border-radius: 12px; 
                padding: 30px;
            }
            QLabel {
                color: #e2e8f0;
            }
        """)
        card_layout = QVBoxLayout(card)
        card_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.setSpacing(12)
        
        logo = QLabel("✨")
        logo.setStyleSheet("font-size: 64px;")
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(logo)
        
        title = QLabel("NovaEnhance AI Studio")
        title.setStyleSheet("font-size: 26px; font-weight: bold; color: #ffffff;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(title)
        
        version = QLabel(f"Version {CURRENT_VERSION} Pro")
        version.setStyleSheet("font-size: 13px; color: #3b82f6; font-weight: 600; margin-top: -10px;")
        version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(version)
        
        desc = QLabel(
            "NovaEnhance AI Studio is a professional, high-performance image and "
            "video enhancement suite designed to upscale, sharpen, and restore media files.\n\n"
            "Using advanced state-of-the-art architectures (Real-ESRGAN, GFPGAN, RIFE), "
            "it delivers outstanding detail recovery, face restoration, and fluid motion interpolation."
        )
        desc.setStyleSheet("font-size: 13px; color: #a0aec0; line-height: 150%;")
        desc.setWordWrap(True)
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(desc)
        
        divider = QFrame()
        divider.setFixedHeight(1)
        divider.setStyleSheet("background-color: #1e2030; margin: 10px 0;")
        card_layout.addWidget(divider)
        
        tech_lbl = QLabel("Core Technologies & Engines:")
        tech_lbl.setStyleSheet("font-weight: bold; color: #ffffff; font-size: 12px;")
        tech_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(tech_lbl)
        
        techs = QLabel("PyQt6 • PyTorch / ONNX Runtime • OpenCV • FFmpeg Engine")
        techs.setStyleSheet("color: #718096; font-size: 12px;")
        techs.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(techs)
        
        # Check for Updates Button
        self.btn_update = QPushButton("🔄 Check for Updates")
        self.btn_update.setStyleSheet("""
            QPushButton {
                background-color: #1e293b;
                color: #3b82f6;
                border: 1px solid #3b82f6;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 12px;
                margin-top: 10px;
            }
            QPushButton:hover {
                background-color: #3b82f6;
                color: white;
            }
            QPushButton:disabled {
                border-color: #334155;
                color: #64748b;
            }
        """)
        self.btn_update.clicked.connect(self.check_for_updates)
        card_layout.addWidget(self.btn_update)
        
        copyright_lbl = QLabel("© 2026 NovaEnhance AI Studio. All rights reserved.")
        copyright_lbl.setStyleSheet("color: #4a5568; font-size: 11px; margin-top: 15px;")
        copyright_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(copyright_lbl)
        
        layout.addWidget(card)
        
        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)

    def check_for_updates(self):
        self.btn_update.setEnabled(False)
        self.btn_update.setText("Checking for updates...")
        
        self.update_thread = UpdateCheckerThread()
        self.update_thread.finished.connect(self.on_update_check_finished)
        self.update_thread.start()

    def on_update_check_finished(self, has_update: bool, latest_version: str, url: str, notes: str):
        self.btn_update.setEnabled(True)
        self.btn_update.setText("🔄 Check for Updates")

        if has_update:
            dlg = UpdateDialog(
                self,
                latest_version=latest_version,
                download_url=url,
                notes=notes,
            )
            dlg.exec()
        else:
            QMessageBox.information(
                self, "Software Up to Date",
                f"You are running the latest version of Nova Ultimate Suite (v{CURRENT_VERSION})."
            )
