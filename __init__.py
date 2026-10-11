from .core import (
    INPUT_DIR, OUTPUT_DIR, TEMP_DIR, SETTINGS_FILE, SUPPORTED_EXTENSIONS, SKIP_DIRS,
    EXT_TO_LANGUAGE, AUDIO_SAMPLE_RATE, EXPORT_FORMATS, ENCODERS,
    StudioError, RenderingError, AudioGenerationError, ExportError, FFmpegError,
    log, _format_eta, _subprocess_startup_info, temp_directory,
    Theme, ThemeRegistry, THEMES
)
from .text import (
    Tokenizer, KeyboardOverlay, KEYBOARD_LAYOUTS, TypingProfile, TypingAnimator
)
from .audio import (
    DSP, _compute_lufs, _normalize_to_lufs,
    Voice, SoundRenderer, VoiceFactory, SimpleSoundGen, SOUND_PRESETS, CLICK_DURATIONS
)
from .rendering import CodeRenderer, MultiPaneRenderer
from .export import ExportConfig, ExportConfigBuilder, AudioWorker, VideoExporter

# Re-export PySide6 elements used in the main project UIs
from PySide6.QtCore import Qt, QTime, QThread, Signal, Slot, QObject, QTimer
from PySide6.QtGui import QColor, QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, 
    QGroupBox, QLabel, QPushButton, QCheckBox, QComboBox, QSpinBox, QDoubleSpinBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog, QMessageBox, QProgressBar,
    QTimeEdit, QGridLayout, QFrame, QLineEdit
)
