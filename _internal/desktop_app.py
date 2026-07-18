"""
desktop_app.py — Nova Ultimate Suite
Main window orchestrator.  Kept at the project root so the PyInstaller spec
file (Nova_Ultimate_Suite.spec) and run.py do not need any changes.

All business logic, theming, icon utilities, hardware detection, cookie
management and the downloader tab have been extracted to focused modules:

    core/logging_setup.py   — logging configuration
    core/cuda_init.py       — GPU/CUDA DLL bootstrap
    core/exceptions.py      — global exception hooks
    core/system_info.py     — hardware info, URL & platform helpers
    core/cookie_utils.py    — shared Netscape cookie helpers
    ui/theme.py             — QSS stylesheet
    ui/icon_utils.py        — SVG icon generators & Qt render helpers
    ui/widgets/             — HoverSvgLabel, create_centered_checkbox
    ui/downloader_page.py   — complete Nova Downloader tab widget
"""
import os
import sys

# Optimize Intel MKL math library performance on AMD processors
os.environ.setdefault("MKL_DEBUG_CPU_TYPE", "5")

from version import __version__, APP_NAME

# ── Initialize QApplication and QSplashScreen instantly to show startup progress ──
from PyQt6.QtWidgets import QApplication, QSplashScreen, QProgressBar, QVBoxLayout
from PyQt6.QtGui import QPixmap, QColor
from PyQt6.QtCore import Qt

qt_app = QApplication.instance()
if not qt_app:
    qt_app = QApplication(sys.argv)

splash = None
current_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(current_dir, "Nova Ultimate Suite.png")
if os.path.exists(logo_path):
    try:
        pixmap = QPixmap(logo_path)
        if not pixmap.isNull():
            pixmap = pixmap.scaled(500, 500, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            splash = QSplashScreen(pixmap, Qt.WindowType.WindowStaysOnTopHint)
            
            # Add a progress bar overlay to the splash screen
            progress_bar = QProgressBar(splash)
            progress_bar.setRange(0, 100)
            progress_bar.setValue(0)
            progress_bar.setTextVisible(True)
            progress_bar.setStyleSheet("""
                QProgressBar {
                    background-color: #1e293b;
                    color: #ffffff;
                    border: 1px solid #475569;
                    border-radius: 4px;
                    text-align: center;
                    font-weight: bold;
                    font-size: 10px;
                }
                QProgressBar::chunk {
                    background-color: #3b82f6;
                    border-radius: 3px;
                }
            """)
            progress_bar.setGeometry(10, 470, 480, 20)
            
            splash.show()
            splash.showMessage("Initializing Nova Ultimate Suite...", Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter, QColor("white"))
            qt_app.processEvents()
    except Exception as e:
        print(f"Failed to show splash screen: {e}")

def update_splash(progress, message):
    if splash:
        try:
            splash.showMessage(message, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter, QColor("white"))
            for child in splash.children():
                if isinstance(child, QProgressBar):
                    child.setValue(progress)
                    break
            qt_app.processEvents()
        except Exception:
            pass

update_splash(10, "Loading logger configurations...")
import shutil
import logging

# ── 1. Logging must be the very first thing ──────────────────────────────────
from core.logging_setup import setup_logging
logger = setup_logging()

update_splash(20, "Initializing CUDA DLL bindings...")
# ── 2. CUDA / GPU DLL paths ──────────────────────────────────────────────────
from core.cuda_init import init_cuda_dlls
init_cuda_dlls()

update_splash(30, "Loading user interface dependencies...")

# ── 3. Qt imports ────────────────────────────────────────────────────────────
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QButtonGroup, QStackedWidget, QFrame,
    QGroupBox, QGridLayout, QTabWidget, QMessageBox,
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QIcon

update_splash(40, "Configuring exceptions handler...")
# ── 4. Project modules ───────────────────────────────────────────────────────
from core.exceptions import exception_hook, thread_exception_hook

update_splash(50, "Loading styling themes...")
from ui.theme import get_stylesheet

update_splash(60, "Configuring vector icons module...")
from ui.icon_utils import (
    create_vector_icon,
    get_tab_icon_svg,
    render_svg_to_pixmap,
)

update_splash(70, "Initializing downloader components...")
from ui.downloader_page import DownloaderPage
from app.downloader import get_settings, save_settings, get_download_dir


# ── Main window ──────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    """
    Thin orchestrator that owns the top navigation and the QStackedWidget.
    Each tab content lives in its own QWidget subclass.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} V{__version__} (Licensed)")

        # Restore saved window geometry
        settings = get_settings()
        default_w, default_h = 1500, 938
        saved_w = settings.get("window_width", default_w)
        saved_h = settings.get("window_height", default_h)
        is_maximized = settings.get("window_maximized", False)

        self.resize(saved_w, saved_h)

        screen = QApplication.primaryScreen()
        if screen:
            sg = screen.geometry()
            self.move(
                max(0, (sg.width()  - saved_w) // 2),
                max(0, (sg.height() - saved_h) // 2),
            )

        self.setMinimumSize(QSize(1120, 680))
        self.setStyleSheet(get_stylesheet())

        if is_maximized:
            self.showMaximized()

        # Icon / logo paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.logo_path = os.path.join(current_dir, "Nova Ultimate Suite.png")
        self.ico_path  = os.path.join(current_dir, "Nova Ultimate Suite.ico")

        icon_path = self.ico_path if os.path.exists(self.ico_path) else self.logo_path
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
            if os.name == 'nt':
                import ctypes
                try:
                    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                        "nova.ultimate.suite.gemini.v1")
                except Exception:
                    pass

        self.setup_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def setup_ui(self) -> None:
        """Create the top navigation bar and the stacked content area."""
        base_widget = QWidget()
        self.setCentralWidget(base_widget)

        base_layout = QVBoxLayout(base_widget)
        base_layout.setContentsMargins(10, 10, 10, 10)
        base_layout.setSpacing(5)

        # ── Top navigation tabs ────────────────────────────────────────
        header = QHBoxLayout()
        header.setSpacing(2)

        def _tab_btn(label: str, icon_name: str) -> QPushButton:
            btn = QPushButton(f" {label}")
            btn.setIcon(create_vector_icon(get_tab_icon_svg(icon_name, "#00a2ff")))
            btn.setIconSize(QSize(16, 16))
            btn.setCheckable(True)
            btn.setProperty("class", "header-tab-btn")
            return btn

        self.btn_tab_downloader = _tab_btn("Nova Downloader",  "downloader")
        self.btn_tab_converter  = _tab_btn("NovaEnhance",      "converter")
        self.btn_tab_editor     = _tab_btn("Nova Video Studio", "editor")
        self.btn_tab_queue      = _tab_btn("Render Queue",      "queue")
        self.btn_tab_tools      = _tab_btn("Merger & Splitter", "tools")
        self.btn_tab_dubber     = _tab_btn("Video Dubber AI",   "dubber")
        self.btn_tab_settings   = _tab_btn("Settings",          "settings")

        self.btn_tab_downloader.setChecked(True)

        self.header_tab_group = QButtonGroup(self)
        for btn in [self.btn_tab_downloader, self.btn_tab_converter, self.btn_tab_editor,
                    self.btn_tab_queue, self.btn_tab_tools, self.btn_tab_dubber,
                    self.btn_tab_settings]:
            self.header_tab_group.addButton(btn)
            header.addWidget(btn)
        header.addStretch()
        base_layout.addLayout(header)

        # ── Stacked pages ─────────────────────────────────────────────
        self.content_stack = QStackedWidget()
        base_layout.addWidget(self.content_stack)

        update_splash(75, "Loading Downloader tab...")
        self._setup_page_downloader()   # index 0
        update_splash(80, "Loading Video Enhancer tab...")
        self._setup_page_converter()    # index 1
        update_splash(84, "Loading Video Editor tab...")
        self._setup_page_editor()       # index 2
        update_splash(88, "Loading Batch Queue tab...")
        self._setup_page_queue()        # index 3
        update_splash(92, "Loading Merger & Splitter tools...")
        self._setup_page_tools()        # index 4
        update_splash(96, "Loading AI Video Dubber tab...")
        self._setup_page_dubber()       # index 5
        update_splash(99, "Loading Settings & Diagnostics...")
        self._setup_page_diagnostics()  # index 6

        # Connect navigation
        self.btn_tab_downloader.clicked.connect(lambda: self.content_stack.setCurrentIndex(0))
        self.btn_tab_converter.clicked.connect( lambda: self.content_stack.setCurrentIndex(1))
        self.btn_tab_editor.clicked.connect(    lambda: self.content_stack.setCurrentIndex(2))
        self.btn_tab_queue.clicked.connect(     lambda: self.content_stack.setCurrentIndex(3))
        self.btn_tab_tools.clicked.connect(     lambda: self.content_stack.setCurrentIndex(4))
        self.btn_tab_dubber.clicked.connect(    lambda: self.content_stack.setCurrentIndex(5))
        self.btn_tab_settings.clicked.connect(
            lambda: [self.content_stack.setCurrentIndex(6), self.update_settings_stats()])

    # ------------------------------------------------------------------
    # Page factories
    # ------------------------------------------------------------------

    def _setup_page_downloader(self) -> None:
        """Index 0 — full Nova Downloader tab (owns its own timers)."""
        self.downloader_page = DownloaderPage(self)
        self.content_stack.addWidget(self.downloader_page)

    def _setup_page_converter(self) -> None:
        """Index 1 — NovaEnhance AI upscaler / converter."""
        from ui.dashboard import ConverterDashboard
        self.content_stack.addWidget(ConverterDashboard(self))

    def _setup_page_editor(self) -> None:
        """Index 2 — Nova Video Studio (timeline editor)."""
        from src.ui.editor_page import EditorPage
        self.content_stack.addWidget(EditorPage(self))

    def _setup_page_queue(self) -> None:
        """Index 3 — Render Queue (batch processing)."""
        from src.ui.batch_queue_dialog import BatchQueueDialog
        self.batch_render_queue_dialog = BatchQueueDialog(self, is_embedded=True)
        self.content_stack.addWidget(self.batch_render_queue_dialog)

    def _setup_page_tools(self) -> None:
        """Index 4 — Merger & Splitter tools."""
        from ui.merger_splitter import MergerSplitterPage
        self.merger_splitter_page = MergerSplitterPage(self)
        self.content_stack.addWidget(self.merger_splitter_page)

    def _setup_page_dubber(self) -> None:
        """Index 5 — AI Video Dubber."""
        from ui.video_dubber import VideoDubberPage
        self.video_dubber_page = VideoDubberPage(self)
        self.content_stack.addWidget(self.video_dubber_page)

    def _setup_page_diagnostics(self) -> None:
        """Index 6 — System Settings & Diagnostics."""
        from services.gpu_manager import GPUManager
        self.gpu_manager = GPUManager()

        page = QFrame()
        page.setProperty("class", "content-page")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)

        title_lbl = QLabel("🛠️ System Settings")
        title_lbl.setStyleSheet("font-size: 20px; font-weight: bold; color: #00a2ff;")
        layout.addWidget(title_lbl)

        settings_tabs = QTabWidget()
        settings_tabs.setObjectName("settings-tabs")

        # System & Storage tab
        tab_system = QWidget()
        sys_layout = QVBoxLayout(tab_system)
        sys_layout.setSpacing(15)

        grid = QGridLayout()
        grid.setSpacing(15)

        # Diagnostics box
        diag_box = QGroupBox("System & Binary Status")
        diag_layout = QVBoxLayout(diag_box)
        diag_layout.setSpacing(8)

        for label_text, attr_name in [
            ("FFmpeg Binaries:",  "lbl_ffmpeg_status"),
            ("FFprobe Binaries:", "lbl_ffprobe_status"),
        ]:
            row = QHBoxLayout()
            row.addWidget(QLabel(label_text))
            lbl = QLabel("Checking...")
            setattr(self, attr_name, lbl)
            row.addStretch()
            row.addWidget(lbl)
            diag_layout.addLayout(row)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("App Mode:"))
        app_mode = (
            "Compiled/Frozen (EXE)"
            if getattr(sys, 'frozen', False)
            else "Python Source Developer"
        )
        lbl_mode = QLabel(app_mode)
        lbl_mode.setStyleSheet("color: #00a2ff; font-weight: bold;")
        mode_row.addStretch()
        mode_row.addWidget(lbl_mode)
        diag_layout.addLayout(mode_row)
        grid.addWidget(diag_box, 0, 0)

        # Storage stats box
        stats_box = QGroupBox("Storage & Library Stats")
        stats_layout = QVBoxLayout(stats_box)
        stats_layout.setSpacing(8)

        for label_text, attr_name, default_val in [
            ("Disk Free Space:",   "lbl_free_space",  "Calculating..."),
            ("Saved Videos:",      "lbl_total_saved", "0 videos"),
            ("Library Disk Size:", "lbl_total_size",  "0 B"),
        ]:
            row = QHBoxLayout()
            row.addWidget(QLabel(label_text))
            lbl = QLabel(default_val)
            setattr(self, attr_name, lbl)
            row.addStretch()
            row.addWidget(lbl)
            stats_layout.addLayout(row)

        grid.addWidget(stats_box, 0, 1)
        sys_layout.addLayout(grid)

        btn_refresh = QPushButton("🔄 Refresh System Information")
        btn_refresh.setProperty("class", "btn-blue")
        btn_refresh.setFixedWidth(240)
        btn_refresh.clicked.connect(self.update_settings_stats)
        sys_layout.addWidget(btn_refresh, 0, Qt.AlignmentFlag.AlignCenter)
        sys_layout.addStretch()

        settings_tabs.addTab(tab_system, "🖥️ System & Storage")
        layout.addWidget(settings_tabs)
        self.content_stack.addWidget(page)

    # ------------------------------------------------------------------
    # Cross-page compatibility stubs
    # (called by DownloaderPage and other sub-pages via main_window ref)
    # ------------------------------------------------------------------

    def update_global_status(self, message: str, progress_val: int = 0) -> None:
        """Compatibility stub so sub-pages can call dashboard.update_global_status()."""
        logger.info(f"[Status] {message}")

    def refresh_library(self) -> None:
        """Compatibility stub — library scanning is owned by DownloaderPage."""
        pass

    def update_settings_stats(self) -> None:
        """Refresh FFmpeg / disk-space labels on the Settings tab asynchronously to prevent GUI freezes."""
        ffmpeg_found  = shutil.which("ffmpeg")  is not None
        ffprobe_found = shutil.which("ffprobe") is not None

        if hasattr(self, 'lbl_ffmpeg_status'):
            self.lbl_ffmpeg_status.setText(
                "Installed (Available)" if ffmpeg_found else "Not Found (Transcoding Disabled)")
            self.lbl_ffmpeg_status.setStyleSheet(
                "color: #2e7d32; font-weight: bold;"
                if ffmpeg_found else "color: #c62828; font-weight: bold;")

        if hasattr(self, 'lbl_ffprobe_status'):
            self.lbl_ffprobe_status.setText(
                "Installed (Available)" if ffprobe_found else "Not Found (Diagnostics Disabled)")
            self.lbl_ffprobe_status.setStyleSheet(
                "color: #2e7d32; font-weight: bold;"
                if ffprobe_found else "color: #c62828; font-weight: bold;")

        settings = get_settings()
        download_dir = get_download_dir()
        if not os.path.exists(download_dir):
            return

        try:
            _, _, free = shutil.disk_usage(download_dir)
            if hasattr(self, 'lbl_free_space'):
                self.lbl_free_space.setText(f"{free / (1024**3):.2f} GB")
        except Exception:
            pass

        # Traversal of directory tree can take seconds on slow disks; run in background thread
        import threading
        
        def scan_worker():
            try:
                scan_dirs = [download_dir]
                for key in ("video_dir", "photo_dir"):
                    d = settings.get(key)
                    if d and os.path.exists(d):
                        scan_dirs.append(d)

                file_count = 0
                total_bytes = 0
                seen = set()
                video_exts = ('.mp4', '.mkv', '.webm', '.avi', '.mov')
                skip_tokens = ('.temp', '.part', '.ytdl', 'temp_transcode_')

                for s_dir in scan_dirs:
                    if not os.path.exists(s_dir):
                        continue
                    for root, _, filenames in os.walk(s_dir):
                        for name in filenames:
                            path = os.path.abspath(os.path.join(root, name))
                            if path in seen:
                                continue
                            seen.add(path)
                            name_lower = name.lower()
                            if name_lower.endswith(video_exts) and \
                                    not any(t in name_lower for t in skip_tokens):
                                file_count += 1
                                total_bytes += os.path.getsize(path)

                size_f = float(total_bytes)
                for unit in ['B', 'KB', 'MB', 'GB']:
                    if size_f < 1024.0:
                        size_str = f"{size_f:.2f} {unit}"
                        break
                    size_f /= 1024.0
                else:
                    size_str = f"{size_f:.2f} GB"

                # Update labels in the main thread thread-safely
                from PyQt6.QtCore import QMetaObject, Q_ARG
                if hasattr(self, 'lbl_total_saved'):
                    QMetaObject.invokeMethod(
                        self.lbl_total_saved, "setText",
                        Qt.ConnectionType.QueuedConnection, Q_ARG(str, f"{file_count} videos")
                    )
                if hasattr(self, 'lbl_total_size'):
                    QMetaObject.invokeMethod(
                        self.lbl_total_size, "setText",
                        Qt.ConnectionType.QueuedConnection, Q_ARG(str, size_str)
                    )
            except Exception as scan_err:
                logger.error("Failed to run background stats scan: %s", scan_err)

        threading.Thread(target=scan_worker, daemon=True).start()

    # ------------------------------------------------------------------
    # File management helpers (called from library / downloader pages)
    # ------------------------------------------------------------------

    def play_video(self, file_path: str) -> None:
        if not os.path.exists(file_path):
            self.show_message("Error", "File has been moved or deleted.")
            return
        try:
            os.startfile(file_path)
        except Exception as e:
            self.show_message("Error", f"Failed to play video: {e}")

    def show_in_folder(self, file_path: str) -> None:
        import subprocess
        if not os.path.exists(file_path):
            self.show_message("Error", "File does not exist.")
            return
        try:
            subprocess.run(["explorer", "/select,", os.path.normpath(file_path)])
        except Exception as e:
            self.show_message("Error", f"Could not open containing folder: {e}")

    def delete_video(self, file_path: str, title: str) -> None:
        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Are you sure you want to delete this video?\n\n{title}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                self.show_toast("Deleted", "File deleted successfully.")
                self.update_settings_stats()
            except Exception as e:
                self.show_message("Error", f"Failed to delete file:\n{e}")

    # ------------------------------------------------------------------
    # UI message helpers
    # ------------------------------------------------------------------

    def show_message(self, title: str, text: str) -> None:
        QMessageBox.information(self, title, text)

    def show_toast(self, title: str, message: str) -> None:
        self.statusBar().showMessage(f"[*] {title}: {message}", 5000)

    # ------------------------------------------------------------------
    # Window lifecycle
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802
        # Persist window geometry
        try:
            settings = get_settings()
            is_maximized = self.isMaximized()
            settings["window_maximized"] = is_maximized
            geom = self.normalGeometry() if is_maximized else (
                self.geometry() if not self.isMinimized() else None
            )
            if geom and geom.width() > 100 and geom.height() > 100:
                settings["window_width"]  = geom.width()
                settings["window_height"] = geom.height()
                settings["window_x"]      = geom.x()
                settings["window_y"]      = geom.y()
            save_settings(settings)
        except Exception as e:
            logger.error(f"Failed to save window state on close: {e}")

        # Close any embedded batch dialog
        if hasattr(self, 'batch_render_queue_dialog') and \
                self.batch_render_queue_dialog is not None:
            self.batch_render_queue_dialog._force_close = True
            try:
                self.batch_render_queue_dialog.close()
            except Exception:
                pass

        # Cleanup local VoxCPM server processes if any were started
        try:
            from app.voxcpm_manager import LocalVoxCPMServerManager
            LocalVoxCPMServerManager.cleanup_all_servers()
        except Exception as e:
            logger.error(f"Failed to cleanup local VoxCPM server processes: {e}")

        event.accept()
        QApplication.quit()


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    import threading
    threading.excepthook = thread_exception_hook

    from PyQt6.QtNetwork import QLocalServer, QLocalSocket

    server_name = "nova_ultimate_suite_single_instance_lock"
    
    global qt_app, splash

    # Single-instance guard
    socket = QLocalSocket()
    socket.connectToServer(server_name)
    if socket.waitForConnected(500):
        socket.write(b"ACTIVATE")
        socket.waitForBytesWritten(1000)
        socket.disconnectFromServer()
        logger.info("Another instance is already running. Activating and exiting.")
        if splash:
            splash.close()
        sys.exit(0)

    QLocalServer.removeServer(server_name)
    server = QLocalServer()
    if not server.listen(server_name):
        logger.warning(f"Could not start local server: {server.errorString()}")

    window = None

    def handle_new_connection() -> None:
        nonlocal window
        client = server.nextPendingConnection()
        if client:
            if client.waitForReadyRead(1000):
                msg = client.readAll().data()
                if msg == b"ACTIVATE" and window:
                    if window.isMinimized():
                        window.showNormal()
                    else:
                        window.show()
                    window.raise_()
                    window.activateWindow()
            client.disconnectFromServer()

    server.newConnection.connect(handle_new_connection)

    if hasattr(Qt, "HighDpiScaleFactorRoundingPolicy"):
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    # Trigger background yt-dlp update check
    try:
        from app.downloader import check_and_update_ytdlp
        check_and_update_ytdlp(force=False)
    except Exception as e:
        logger.error(f"Failed to trigger startup yt-dlp update check: {e}")

    window = MainWindow()
    window.show()
    if splash:
        splash.finish(window)

    sys.excepthook = exception_hook

    # ── Auto-check for updates 5 s after startup ──────────────────────────────
    def _auto_update_check():
        try:
            from services.update_service import UpdateCheckerThread
            from ui.update_dialog import UpdateDialog, get_skipped_version

            def _on_result(has_update, latest_version, url, notes):
                if not has_update:
                    return
                if get_skipped_version() == latest_version:
                    return  # User already skipped this version
                dlg = UpdateDialog(window, latest_version=latest_version,
                                   download_url=url, notes=notes)
                dlg.exec()

            _checker = UpdateCheckerThread()
            _checker.finished.connect(_on_result)
            _checker.start()
            # Keep reference alive
            window._startup_update_thread = _checker
        except Exception as e:
            logger.warning(f"Startup update check failed: {e}")

    from PyQt6.QtCore import QTimer as _QTimer
    _QTimer.singleShot(5000, _auto_update_check)
    # ─────────────────────────────────────────────────────────────────────────

    exit_code = qt_app.exec()

    server.close()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

# Verification comment for incremental rebuild
