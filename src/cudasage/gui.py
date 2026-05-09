"""PyQt6 desktop GUI for cuda-sage.

This module provides a native desktop interface for users who prefer not to use
CLI commands directly.
"""

from __future__ import annotations

import html
import json
import os
import sys
from pathlib import Path

from .analyzers.divergence import DivergenceAnalyzer
from .analyzers.memory import MemoryAnalyzer
from .analyzers.occupancy import OccupancyAnalyzer
from .models.architectures import ARCHITECTURES, get_arch
from .parsers.ptx_parser import PTXParser
from .reporter import build_json_report

_THEME_EXTRA = {
    "danger": "#ff6b6b",
    "warning": "#ffb84d",
    "success": "#2ed573",
    "font_family": "DejaVu Sans",
    "density_scale": "-1",
}

_LIMITER_EXPLANATIONS = {
    "registers": "Register pressure is capping how many warps can stay resident on each SM.",
    "shared_memory": "Per-block shared memory usage is limiting the number of concurrent blocks.",
    "threads_per_block": "The chosen block size is the current occupancy bottleneck.",
    "hw_block_limit": "Hardware block-per-SM limits are the dominant constraint for this launch shape.",
    "threads_per_block_invalid": "The assumed block size is invalid for the selected architecture.",
    "arch_invalid": "The selected architecture model returned invalid hardware limits.",
}

_METRIC_TOOLTIPS = {
    "Kernel": "The PTX entry name being analyzed.",
    "Occupancy %": "Estimated fraction of available warps resident on an SM for the chosen launch shape.",
    "Limiter": "The resource currently preventing higher occupancy.",
    "Regs": "Estimated 32-bit registers used per thread. Higher values often reduce active warps.",
    "Spills": "Local-memory load/store operations caused by register pressure. Higher is usually bad.",
    "Divergence": "Detected branches whose predicates depend on thread-varying values.",
    "Missing Sync": "Whether shared-memory writes appear to be used without an observed barrier.",
}


def _require_pyqt6() -> dict[str, object]:
    try:
        from PyQt6.QtCore import Qt, QTimer
        from PyQt6.QtWidgets import (
            QApplication,
            QAbstractItemView,
            QCheckBox,
            QComboBox,
            QFrame,
            QFileDialog,
            QFormLayout,
            QGridLayout,
            QGroupBox,
            QHBoxLayout,
            QHeaderView,
            QLabel,
            QLineEdit,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QSizePolicy,
            QSplitter,
            QSpinBox,
            QTableWidget,
            QTableWidgetItem,
            QTabWidget,
            QTextEdit,
            QVBoxLayout,
            QWidget,
        )
    except ImportError as exc:
        raise SystemExit(
            "PyQt6 is required for the desktop GUI. Install with: pip install -e '.[gui]'"
        ) from exc

    return {
        "Qt": Qt,
        "QTimer": QTimer,
        "QApplication": QApplication,
        "QAbstractItemView": QAbstractItemView,
        "QCheckBox": QCheckBox,
        "QComboBox": QComboBox,
        "QFrame": QFrame,
        "QFileDialog": QFileDialog,
        "QFormLayout": QFormLayout,
        "QGridLayout": QGridLayout,
        "QGroupBox": QGroupBox,
        "QHBoxLayout": QHBoxLayout,
        "QHeaderView": QHeaderView,
        "QLabel": QLabel,
        "QLineEdit": QLineEdit,
        "QMainWindow": QMainWindow,
        "QMessageBox": QMessageBox,
        "QPushButton": QPushButton,
        "QSizePolicy": QSizePolicy,
        "QSplitter": QSplitter,
        "QSpinBox": QSpinBox,
        "QTableWidget": QTableWidget,
        "QTableWidgetItem": QTableWidgetItem,
        "QTabWidget": QTabWidget,
        "QTextEdit": QTextEdit,
        "QVBoxLayout": QVBoxLayout,
        "QWidget": QWidget,
    }


def _load_qt_material() -> dict[str, object] | None:
    try:
        from qt_material import apply_stylesheet, list_themes
    except ImportError:
        return None
    themes = list_themes()
    return {"apply_stylesheet": apply_stylesheet, "themes": themes}


def _custom_stylesheet() -> str:
    return """
QWidget#centralShell {
    background: transparent;
}

QFrame#heroPanel {
    background: qlineargradient(
        x1: 0, y1: 0, x2: 1, y2: 1,
        stop: 0 rgba(19, 34, 56, 0.96),
        stop: 0.55 rgba(12, 87, 121, 0.92),
        stop: 1 rgba(36, 182, 164, 0.88)
    );
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 28px;
}

QFrame#panel,
QFrame#statCard {
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 22px;
    background: rgba(255, 255, 255, 0.035);
}

QLabel#eyebrow {
    color: rgba(255, 255, 255, 0.82);
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
}

QLabel#heroTitle {
    color: white;
    font-size: 30px;
    font-weight: 700;
}

QLabel#heroSubtitle,
QLabel#panelSubtitle,
QLabel#statCaption {
    color: rgba(255, 255, 255, 0.76);
    font-size: 13px;
}

QLabel#panelTitle {
    font-size: 19px;
    font-weight: 650;
}

QLabel#statValue {
    font-size: 28px;
    font-weight: 700;
}

QPushButton {
    min-height: 40px;
    border-radius: 14px;
    padding: 8px 16px;
    font-weight: 600;
    text-transform: none;
}

QLineEdit,
QComboBox,
QSpinBox,
QTextEdit,
QTableWidget,
QTabWidget::pane {
    border-radius: 16px;
}

QLineEdit,
QComboBox,
QSpinBox {
    min-height: 42px;
    padding: 8px 12px;
}

QTabWidget::pane {
    margin-top: 8px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    background: rgba(255, 255, 255, 0.03);
}

QTabBar::tab {
    min-width: 108px;
    padding: 12px 18px;
    margin-right: 6px;
    border-top-left-radius: 14px;
    border-top-right-radius: 14px;
}

QTextEdit#reportPane,
QTextEdit#jsonPane,
QTextEdit#diffPane {
    padding: 14px;
}

QTableWidget {
    gridline-color: transparent;
    alternate-background-color: rgba(255, 255, 255, 0.025);
    selection-background-color: rgba(46, 213, 115, 0.2);
}

QHeaderView::section {
    border: 0;
    padding: 10px 12px;
    font-weight: 700;
}
"""


def _fallback_stylesheet() -> str:
    return """
QMainWindow {
    background-color: #0f1722;
    color: #f7fbff;
}

QWidget {
    color: #f7fbff;
}

QFrame#heroPanel {
    background-color: #12334b;
    border: 1px solid #1c5b7f;
    border-radius: 28px;
}

QFrame#panel,
QFrame#statCard,
QTextEdit,
QTableWidget,
QLineEdit,
QComboBox,
QSpinBox {
    background-color: #142130;
    border: 1px solid #294055;
    border-radius: 16px;
}

QPushButton {
    background-color: #1f6feb;
    border-radius: 14px;
    padding: 10px 16px;
    font-weight: 600;
}
"""


def _apply_app_theme(app: object, theme_support: dict[str, object] | None, theme_name: str) -> None:
    app.setStyle("Fusion")
    if theme_support:
        theme_support["apply_stylesheet"](
            app,
            theme=theme_name,
            invert_secondary=theme_name.startswith("light_"),
            extra=_THEME_EXTRA,
        )
        app.setStyleSheet(app.styleSheet() + "\n" + _custom_stylesheet())
        return
    app.setStyleSheet(_fallback_stylesheet())


def _html_list(items: list[str]) -> str:
    if not items:
        return "<p>No suggestions for this slice.</p>"
    return "<ul>" + "".join(f"<li>{html.escape(item)}</li>" for item in items) + "</ul>"


def _kernel_priority(report: dict) -> tuple[str, str]:
    occ = report["occupancy"]["percent"]
    div = report["divergence"]
    mem = report["memory"]
    if mem["possible_missing_sync"] or div["high_severity_count"] > 0 or mem["spill_ops"] > 10:
        return "High", "#ff6b6b"
    if occ < 50 or div["site_count"] > 0 or mem["spill_ops"] > 0:
        return "Medium", "#ffd166"
    return "Low", "#2ed573"


def _occupancy_meaning(percent: float) -> str:
    if percent < 25:
        return "Very low occupancy. The GPU may struggle to hide latency, so throughput can become highly sensitive to memory stalls."
    if percent < 50:
        return "Moderate occupancy. The kernel may still perform well, but latency hiding headroom is limited."
    if percent < 75:
        return "Healthy occupancy. Resource usage is allowing a solid level of parallelism on each SM."
    return "High occupancy. Parallelism is unlikely to be the first place to look unless other warnings are severe."


def _primary_takeaway(report: dict) -> str:
    occ = report["occupancy"]
    div = report["divergence"]
    mem = report["memory"]

    if mem["possible_missing_sync"]:
        return "The highest-priority issue is correctness-adjacent: shared-memory writes appear without a visible barrier."
    if mem["spill_ops"] > 0:
        return "Register pressure is material enough to trigger spills, so reducing live values is likely the best first optimization step."
    if div["high_severity_count"] > 0:
        return "The branch structure looks warp-hostile. High-severity divergence usually deserves attention before micro-tuning."
    if occ["limiting_factor"] in _LIMITER_EXPLANATIONS:
        return _LIMITER_EXPLANATIONS[occ["limiting_factor"]]
    return "No dominant risk stands out immediately; focus on validating assumptions with runtime profiling."


def _where_to_tune(report: dict) -> list[str]:
    occ = report["occupancy"]
    mem = report["memory"]
    div = report["divergence"]
    knobs: list[str] = []

    if occ["limiting_factor"] == "registers":
        knobs.append("Tune register pressure: shorten live ranges, split long kernels, or cap registers with launch bounds or maxrregcount.")
    if occ["limiting_factor"] == "shared_memory":
        knobs.append("Tune shared-memory footprint: reduce tile size, shrink staging buffers, or revisit smem/L1 tradeoffs.")
    if occ["limiting_factor"] == "threads_per_block":
        knobs.append("Tune launch shape: compare the current block size against 128, 256, and 512-thread launches.")
    if mem["spill_ops"] > 0:
        knobs.append("Tune local-memory traffic by cutting temporaries and avoiding unnecessary wide live values.")
    if mem["memory_bound_likely"]:
        knobs.append("Tune data movement: increase reuse, fuse stages, or use wider global loads where alignment permits.")
    if div["site_count"] > 0:
        knobs.append("Tune control flow: replace divergent branches with predication or restructure work so warps take the same path.")
    if mem["possible_missing_sync"]:
        knobs.append("Tune synchronization: add __syncthreads() around shared-memory handoff points.")

    if not knobs:
        knobs.append("No single tuning lever dominates. Use runtime metrics to choose between IPC, memory bandwidth, and occupancy work.")
    return knobs


def _suggest_modifications(report: dict) -> list[str]:
    occ = report["occupancy"]
    overview = report["overview"]
    div = report["divergence"]
    mem = report["memory"]
    suggestions: list[str] = []

    if occ["limiting_factor"] == "registers":
        suggestions.append(
            f"Refactor the kernel to reduce per-thread register use from about {overview['registers']} registers, especially around long-lived temporaries."
        )
    if occ["limiting_factor"] == "shared_memory":
        suggestions.append(
            f"Reduce the per-block shared-memory footprint from {overview['shared_mem_bytes']} bytes by shrinking tiles or staging less data at once."
        )
    if occ["limiting_factor"] == "threads_per_block":
        suggestions.append(
            "Benchmark alternate launch sizes - especially 128, 256, and 512 threads per block - and keep the one with the best end-to-end tradeoff."
        )
    if mem["spill_ops"] > 0:
        suggestions.append(
            "Cut spill traffic by reducing temporary variables, breaking the kernel into phases, or moving reused intermediates into shared memory."
        )
    if div["high_severity_count"] > 0:
        suggestions.append(
            "Rewrite odd/even or modulo-based branches so more threads in a warp take the same path, or convert simple branches to predicated selects."
        )
    elif div["site_count"] > 0:
        suggestions.append(
            "Review divergent branches and decide whether they can be hoisted, predicated, or grouped to keep warp behavior more uniform."
        )
    if mem["possible_missing_sync"]:
        suggestions.append(
            "Insert __syncthreads() after shared-memory writes and before dependent reads if the algorithm expects cross-thread reuse."
        )
    if mem["memory_bound_likely"]:
        suggestions.append(
            "Increase arithmetic intensity by reusing loaded data longer, fusing adjacent work, or switching to vectorized loads when alignment is safe."
        )

    if not suggestions:
        suggestions.append(
            "No obvious static fix dominates. Validate branch efficiency, achieved occupancy, and memory throughput with Nsight Compute to choose the next optimization pass."
        )
    return suggestions[:5]


def _metric_explanation_rows(report: dict) -> list[tuple[str, str, str]]:
    occ = report["occupancy"]
    div = report["divergence"]
    mem = report["memory"]
    return [
        (
            "Occupancy",
            f"{occ['percent']}%",
            _occupancy_meaning(occ["percent"]),
        ),
        (
            "Limiter",
            occ["limiting_factor"],
            _LIMITER_EXPLANATIONS.get(occ["limiting_factor"], "This is the main resource currently constraining concurrency."),
        ),
        (
            "Registers",
            str(report["overview"]["registers"]),
            "Registers are allocated per thread. Higher counts can reduce active warps and may eventually cause spills.",
        ),
        (
            "Spills",
            str(mem["spill_ops"]),
            "Spills are local-memory accesses caused by register pressure. They are usually much slower than staying in registers.",
        ),
        (
            "Divergence",
            str(div["site_count"]),
            "Divergence means threads in a warp are taking different control-flow paths, which serializes execution.",
        ),
        (
            "Missing sync",
            str(mem["possible_missing_sync"]),
            "This flag suggests shared-memory data may be consumed without an obvious barrier between writers and readers.",
        ),
        (
            "Arithmetic intensity",
            "n/a" if mem["arithmetic_intensity_proxy"] is None else str(mem["arithmetic_intensity_proxy"]),
            "This approximates how much math the kernel performs per global-memory operation. Low values often indicate a memory-bound kernel.",
        ),
    ]


def _format_metric_explanations_html(report: dict) -> str:
    rows = "".join(
        f"<tr><td><b>{html.escape(name)}</b></td><td>{html.escape(value)}</td><td>{html.escape(meaning)}</td></tr>"
        for name, value, meaning in _metric_explanation_rows(report)
    )
    return (
        "<table cellspacing='0' cellpadding='6' width='100%'>"
        "<thead><tr><th align='left'>Metric</th><th align='left'>Current value</th><th align='left'>What it means</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _format_action_plan_html(report: dict) -> str:
    priority, color = _kernel_priority(report)
    return (
        f"<h2>{html.escape(report['kernel'])} action plan</h2>"
        f"<p><b>Priority:</b> <span style='color:{color}; font-weight:700'>{priority}</span></p>"
        f"<p>{html.escape(_primary_takeaway(report))}</p>"
        f"<h3>What these values mean</h3>"
        f"{_format_metric_explanations_html(report)}"
        f"<h3>Where to tune next</h3>"
        f"{_html_list(_where_to_tune(report))}"
        f"<h3>Suggested modifications now</h3>"
        f"{_html_list(_suggest_modifications(report))}"
        f"<h3>Why the tool is saying this</h3>"
        f"{_html_list(report['occupancy']['suggestions'] + report['divergence']['suggestions'] + report['memory']['suggestions'])}"
    )


def _format_overall_summary_html(reports: list[dict]) -> str:
    if not reports:
        return "<h2>No analysis loaded</h2><p>Open a PTX file and run analysis to get an interpretation and action plan.</p>"

    top_actions: list[str] = []
    for report in reports:
        top_actions.extend(_suggest_modifications(report)[:2])

    deduped_actions: list[str] = []
    for action in top_actions:
        if action not in deduped_actions:
            deduped_actions.append(action)

    return (
        f"<h1>Analysis summary</h1>"
        f"<p>{len(reports)} kernel(s) analyzed. This view explains what the numbers mean and what to change next, instead of only listing raw values.</p>"
        f"<h3>Top recommended modifications</h3>"
        f"{_html_list(deduped_actions[:6])}"
        + "".join(_format_report_html(report) for report in reports)
    )


def _format_report_html(report: dict) -> str:
    occ = report["occupancy"]
    div = report["divergence"]
    mem = report["memory"]
    overview = report["overview"]
    all_suggestions = occ["suggestions"] + div["suggestions"] + mem["suggestions"]
    priority, color = _kernel_priority(report)
    return (
        f"<section>"
        f"<h2>{html.escape(report['kernel'])}</h2>"
        f"<p><b>Priority:</b> <span style='color:{color}; font-weight:700'>{priority}</span></p>"
        f"<p><b>Architecture:</b> {html.escape(report['sm_target'])} | "
        f"<b>Occupancy:</b> {occ['percent']}% | "
        f"<b>Limiter:</b> {html.escape(occ['limiting_factor'])}</p>"
        f"<p><b>Registers:</b> {overview['registers']} | "
        f"<b>Shared memory:</b> {overview['shared_mem_bytes']} bytes | "
        f"<b>Instructions:</b> {overview['instruction_count']}</p>"
        f"<p><b>Divergence sites:</b> {div['site_count']} | "
        f"<b>High severity:</b> {div['high_severity_count']} | "
        f"<b>Spill ops:</b> {mem['spill_ops']}</p>"
        f"<p><b>What this suggests:</b> {html.escape(_primary_takeaway(report))}</p>"
        f"<h3>Where to tune</h3>"
        f"{_html_list(_where_to_tune(report))}"
        f"<h3>Suggested code modifications</h3>"
        f"{_html_list(_suggest_modifications(report))}"
        f"<h3>Recommended follow-up</h3>"
        f"{_html_list(all_suggestions)}"
        f"</section>"
    )


def _format_diff_html(entries: list[dict], baseline_name: str, optimized_name: str, arch_name: str) -> str:
    blocks = [
        f"<h2>{html.escape(baseline_name)} → {html.escape(optimized_name)}</h2>",
        f"<p><b>Architecture:</b> {html.escape(arch_name)}</p>",
    ]
    for entry in entries:
        verdict_color = {
            "IMPROVED": "#2ed573",
            "REGRESSION": "#ff6b6b",
            "NEUTRAL": "#ffd166",
        }[entry["verdict"]]
        blocks.append(
            f"<section>"
            f"<h3>{html.escape(entry['kernel'])}</h3>"
            f"<p><b>Occupancy delta:</b> {entry['occupancy_delta']} | "
            f"<b>Register delta:</b> {entry['register_delta']} | "
            f"<b>Spill delta:</b> {entry['spill_delta']} | "
            f"<b>Divergence delta:</b> {entry['divergence_delta']}</p>"
            f"<p><b>Verdict:</b> <span style='color:{verdict_color}; font-weight:700'>{entry['verdict']}</span></p>"
            f"</section>"
        )
    return "".join(blocks)


def main() -> None:
    qt = _require_pyqt6()
    theme_support = _load_qt_material()

    Qt = qt["Qt"]
    QTimer = qt["QTimer"]
    QApplication = qt["QApplication"]
    QAbstractItemView = qt["QAbstractItemView"]
    QCheckBox = qt["QCheckBox"]
    QComboBox = qt["QComboBox"]
    QFrame = qt["QFrame"]
    QFileDialog = qt["QFileDialog"]
    QFormLayout = qt["QFormLayout"]
    QGridLayout = qt["QGridLayout"]
    QGroupBox = qt["QGroupBox"]
    QHBoxLayout = qt["QHBoxLayout"]
    QHeaderView = qt["QHeaderView"]
    QLabel = qt["QLabel"]
    QLineEdit = qt["QLineEdit"]
    QMainWindow = qt["QMainWindow"]
    QMessageBox = qt["QMessageBox"]
    QPushButton = qt["QPushButton"]
    QSizePolicy = qt["QSizePolicy"]
    QSplitter = qt["QSplitter"]
    QSpinBox = qt["QSpinBox"]
    QTableWidget = qt["QTableWidget"]
    QTableWidgetItem = qt["QTableWidgetItem"]
    QTabWidget = qt["QTabWidget"]
    QTextEdit = qt["QTextEdit"]
    QVBoxLayout = qt["QVBoxLayout"]
    QWidget = qt["QWidget"]

    class MainWindow(QMainWindow):
        def __init__(self, app: object, theme_support: dict[str, object] | None) -> None:
            super().__init__()
            self.app = app
            self.theme_support = theme_support
            self.available_themes = theme_support["themes"] if theme_support else ["built-in"]
            self.current_theme = (
                "dark_teal.xml" if "dark_teal.xml" in self.available_themes else self.available_themes[0]
            )

            self.setWindowTitle("cuda-sage Studio v0.3.1")
            self.resize(1280, 860)

            self.parser = PTXParser()
            self.occ = OccupancyAnalyzer()
            self.div = DivergenceAnalyzer()
            self.mem = MemoryAnalyzer()
            self.last_reports: list[dict] = []
            self.last_diff_entries: list[dict] = []

            self._build_ui()
            self._refresh_summary_metrics([])
            self.guidance_view.setHtml(_format_overall_summary_html([]))

        def _make_panel(self, title: str, subtitle: str) -> tuple[object, object]:
            panel = QFrame()
            panel.setObjectName("panel")
            layout = QVBoxLayout(panel)
            layout.setContentsMargins(20, 20, 20, 20)
            layout.setSpacing(14)

            heading = QLabel(title)
            heading.setObjectName("panelTitle")
            subtitle_label = QLabel(subtitle)
            subtitle_label.setWordWrap(True)
            subtitle_label.setObjectName("panelSubtitle")
            layout.addWidget(heading)
            layout.addWidget(subtitle_label)
            return panel, layout

        def _make_stat_card(self, title: str) -> tuple[object, object]:
            card = QFrame()
            card.setObjectName("statCard")
            layout = QVBoxLayout(card)
            layout.setContentsMargins(16, 14, 16, 14)
            layout.setSpacing(6)

            value = QLabel("-")
            value.setObjectName("statValue")
            caption = QLabel(title)
            caption.setObjectName("statCaption")
            caption.setWordWrap(True)
            layout.addWidget(value)
            layout.addWidget(caption)
            return card, value

        def _build_ui(self) -> None:
            root = QWidget()
            root.setObjectName("centralShell")
            self.setCentralWidget(root)
            layout = QVBoxLayout(root)
            layout.setContentsMargins(22, 22, 22, 22)
            layout.setSpacing(18)

            hero = QFrame()
            hero.setObjectName("heroPanel")
            hero_layout = QVBoxLayout(hero)
            hero_layout.setContentsMargins(24, 24, 24, 24)
            hero_layout.setSpacing(14)

            eyebrow = QLabel("STATIC PTX PERFORMANCE REVIEW")
            eyebrow.setObjectName("eyebrow")
            hero_title = QLabel("cuda-sage Studio")
            hero_title.setObjectName("heroTitle")
            hero_subtitle = QLabel(
                "A polished desktop workspace for occupancy analysis, divergence triage, "
                "memory-risk review, and PTX diffing without living in the terminal.  "
                "\u2014  v0.3.1"
            )
            hero_subtitle.setObjectName("heroSubtitle")
            hero_subtitle.setWordWrap(True)

            theme_row = QHBoxLayout()
            theme_row.setSpacing(10)
            theme_label = QLabel("Theme")
            theme_label.setObjectName("eyebrow")
            self.theme_combo = QComboBox()
            self.theme_combo.setToolTip("Switch the application color theme. Requires qt-material to be installed.")
            self.theme_combo.setAccessibleName("Color theme selector")
            self.theme_combo.setMinimumHeight(42)
            self.theme_combo.setMinimumWidth(180)
            self.theme_combo.addItems(self.available_themes)
            self.theme_combo.setCurrentText(self.current_theme)
            self.theme_combo.currentTextChanged.connect(self._change_theme)
            if not self.theme_support:
                self.theme_combo.setEnabled(False)
                self.theme_combo.setToolTip("Install with pip install -e '.[gui]' to enable qt-material themes.")
            theme_row.addWidget(theme_label)
            theme_row.addWidget(self.theme_combo)
            theme_row.addStretch(1)

            hero_layout.addWidget(eyebrow)
            hero_layout.addWidget(hero_title)
            hero_layout.addWidget(hero_subtitle)
            hero_layout.addLayout(theme_row)

            controls = QSplitter(Qt.Orientation.Horizontal)

            analyze_panel, analyze_layout = self._make_panel(
                "Analyze PTX",
                "Open a kernel dump, set target assumptions, and generate a structured review of bottlenecks and suggestions.",
            )

            self.ptx_path = QLineEdit()
            self.ptx_path.setPlaceholderText("Select a .ptx file to inspect")
            self.ptx_path.setToolTip("Path to the PTX file to analyze. Use Browse to pick a file.")
            self.ptx_path.setAccessibleName("PTX file path")
            self.ptx_path.setMinimumHeight(42)

            pick_ptx_btn = QPushButton("Browse...")
            pick_ptx_btn.setToolTip("Open a file-picker dialog to select a .ptx file (Alt+B).")
            pick_ptx_btn.setAccessibleName("Browse PTX file")
            pick_ptx_btn.setShortcut("Alt+B")
            pick_ptx_btn.setMinimumHeight(40)
            pick_ptx_btn.clicked.connect(self._pick_ptx_file)

            self.arch_combo = QComboBox()
            self.arch_combo.setToolTip("Target GPU architecture. Occupancy limits are read from this model.")
            self.arch_combo.setAccessibleName("Target architecture")
            self.arch_combo.setAccessibleDescription(
                "Select the SM generation that matches the GPU the kernel will run on."
            )
            self.arch_combo.setMinimumHeight(42)
            self.arch_combo.setMinimumWidth(180)
            for sm in sorted(ARCHITECTURES.keys()):
                self.arch_combo.addItem(sm)
            self.arch_combo.setCurrentText("sm_80")

            self.threads_spin = QSpinBox()
            self.threads_spin.setRange(1, 1024)
            self.threads_spin.setSingleStep(32)
            self.threads_spin.setValue(256)
            self.threads_spin.setToolTip(
                "Assumed threads per block for occupancy calculation. Common values: 128, 256, 512."
            )
            self.threads_spin.setAccessibleName("Threads per block")
            self.threads_spin.setMinimumHeight(42)
            self.threads_spin.setMinimumWidth(120)

            self.kernel_filter = QLineEdit()
            self.kernel_filter.setPlaceholderText("Optional kernel name substring")
            self.kernel_filter.setToolTip("Filter results to kernels whose name contains this substring (case-sensitive).")
            self.kernel_filter.setAccessibleName("Kernel filter")
            self.kernel_filter.setAccessibleDescription("Leave empty to analyze all kernels in the file.")
            self.kernel_filter.setMinimumHeight(42)

            self.curve_box = QCheckBox("Include occupancy curve")
            self.curve_box.setToolTip("Emit a CSV of occupancy vs. thread-count alongside the report.")
            self.curve_box.setAccessibleName("Include occupancy curve")

            run_btn = QPushButton("Run Analysis")
            run_btn.setProperty("class", "success")
            run_btn.setToolTip("Parse the selected PTX file and run all analyzers (Alt+R).")
            run_btn.setAccessibleName("Run Analysis")
            run_btn.setAccessibleDescription("Analyze the selected PTX file and populate all result tabs.")
            run_btn.setShortcut("Alt+R")
            run_btn.setMinimumHeight(40)
            run_btn.clicked.connect(self._run_analysis)

            save_json_btn = QPushButton("Save JSON")
            save_json_btn.setToolTip("Export the current analysis results to a JSON file (Alt+S).")
            save_json_btn.setAccessibleName("Save JSON")
            save_json_btn.setShortcut("Alt+S")
            save_json_btn.setMinimumHeight(40)
            save_json_btn.clicked.connect(self._save_json)

            # --- Explicit per-row QHBoxLayouts ---
            # QFormLayout.ExpandingFieldsGrow is overridden by qt-material QSS,
            # causing combos and spinboxes to stay narrow. Using addWidget(w, 1)
            # inside a plain QHBoxLayout is the only approach that is immune to
            # stylesheet interference: stretch factor is a layout property, not
            # a style property, so QSS can never shrink the field back down.

            _LW = 110  # fixed label column width in pixels

            def _lbl(text: str) -> object:
                """Right-aligned label with fixed width to form a consistent label column."""
                lab = QLabel(text)
                lab.setFixedWidth(_LW)
                lab.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                return lab

            def _hrow(*widgets, stretch_idx: int = 0) -> object:
                """One form row: fixed label column + field(s) with explicit stretch."""
                row = QHBoxLayout()
                row.setSpacing(8)
                row.setContentsMargins(0, 0, 0, 0)
                for i, w in enumerate(widgets):
                    row.addWidget(w, 1 if i == stretch_idx else 0)
                return row

            self.arch_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self.threads_spin.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self.kernel_filter.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self.ptx_path.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

            ptx_field_row = _hrow(_lbl("PTX file"), self.ptx_path, stretch_idx=1)
            ptx_button_row = QHBoxLayout()
            ptx_button_row.setSpacing(8)
            ptx_button_row.setContentsMargins(0, 0, 0, 0)
            ptx_button_row.addSpacing(_LW + 8)
            ptx_button_row.addStretch(1)
            ptx_button_row.addWidget(pick_ptx_btn)
            arch_row = _hrow(_lbl("Architecture"), self.arch_combo, stretch_idx=1)
            threads_row = _hrow(_lbl("Threads / block"), self.threads_spin, stretch_idx=1)
            filter_row = _hrow(_lbl("Kernel filter"), self.kernel_filter, stretch_idx=1)

            curve_row = QHBoxLayout()
            curve_row.setSpacing(8)
            curve_row.setContentsMargins(0, 0, 0, 0)
            curve_row.addSpacing(_LW + 8)
            curve_row.addWidget(self.curve_box)
            curve_row.addStretch(1)

            form_vbox = QVBoxLayout()
            form_vbox.setSpacing(10)
            form_vbox.setContentsMargins(0, 0, 0, 0)
            form_vbox.addLayout(ptx_field_row)
            form_vbox.addLayout(ptx_button_row)
            form_vbox.addLayout(arch_row)
            form_vbox.addLayout(threads_row)
            form_vbox.addLayout(filter_row)
            form_vbox.addLayout(curve_row)

            action_row = QHBoxLayout()
            action_row.setSpacing(10)
            action_row.addWidget(run_btn)
            action_row.addWidget(save_json_btn)
            action_row.addStretch(1)
            analyze_layout.addLayout(form_vbox)
            analyze_layout.addLayout(action_row)

            diff_panel, diff_layout_root = self._make_panel(
                "Diff PTX",
                "Compare two PTX artifacts and surface directional changes in occupancy, registers, spills, and divergence.",
            )

            self.base_path = QLineEdit()
            self.base_path.setPlaceholderText("Baseline .ptx")
            self.base_path.setToolTip("Path to the baseline (before-optimisation) PTX file.")
            self.base_path.setAccessibleName("Baseline PTX path")
            self.base_path.setMinimumHeight(42)

            base_btn = QPushButton("Browse baseline...")
            base_btn.setToolTip("Open a file-picker dialog to select the baseline .ptx file.")
            base_btn.setAccessibleName("Browse baseline PTX")
            base_btn.setMinimumHeight(40)
            base_btn.clicked.connect(self._pick_baseline)

            self.opt_path = QLineEdit()
            self.opt_path.setPlaceholderText("Optimized .ptx")
            self.opt_path.setToolTip("Path to the optimized (after-change) PTX file.")
            self.opt_path.setAccessibleName("Optimized PTX path")
            self.opt_path.setMinimumHeight(42)

            opt_btn = QPushButton("Browse optimized...")
            opt_btn.setToolTip("Open a file-picker dialog to select the optimized .ptx file.")
            opt_btn.setAccessibleName("Browse optimized PTX")
            opt_btn.setMinimumHeight(40)
            opt_btn.clicked.connect(self._pick_optimized)

            run_diff_btn = QPushButton("Run Diff")
            run_diff_btn.setProperty("class", "warning")
            run_diff_btn.setToolTip("Compare the two PTX files and populate the Diff tab (Alt+D).")
            run_diff_btn.setAccessibleName("Run Diff")
            run_diff_btn.setShortcut("Alt+D")
            run_diff_btn.setMinimumHeight(40)
            run_diff_btn.clicked.connect(self._run_diff)

            # Diff panel: same explicit-HBox-per-row approach
            self.base_path.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self.opt_path.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

            base_file_row = _hrow(_lbl("Baseline PTX"), self.base_path, stretch_idx=1)
            base_button_row = QHBoxLayout()
            base_button_row.setSpacing(8)
            base_button_row.setContentsMargins(0, 0, 0, 0)
            base_button_row.addSpacing(_LW + 8)
            base_button_row.addStretch(1)
            base_button_row.addWidget(base_btn)

            opt_file_row = _hrow(_lbl("Optimized PTX"), self.opt_path, stretch_idx=1)
            opt_button_row = QHBoxLayout()
            opt_button_row.setSpacing(8)
            opt_button_row.setContentsMargins(0, 0, 0, 0)
            opt_button_row.addSpacing(_LW + 8)
            opt_button_row.addStretch(1)
            opt_button_row.addWidget(opt_btn)

            run_diff_row = QHBoxLayout()
            run_diff_row.addStretch(1)
            run_diff_row.addWidget(run_diff_btn)

            diff_vbox = QVBoxLayout()
            diff_vbox.setSpacing(10)
            diff_vbox.setContentsMargins(0, 0, 0, 0)
            diff_vbox.addLayout(base_file_row)
            diff_vbox.addLayout(base_button_row)
            diff_vbox.addLayout(opt_file_row)
            diff_vbox.addLayout(opt_button_row)
            diff_vbox.addLayout(run_diff_row)

            diff_layout_root.addLayout(diff_vbox)

            controls.addWidget(analyze_panel)
            controls.addWidget(diff_panel)
            controls.setChildrenCollapsible(False)
            controls.setSizes([730, 520])

            self.tabs = QTabWidget()
            self.tabs.setDocumentMode(True)
            self.tabs.setUsesScrollButtons(True)

            summary_tab = QWidget()
            summary_layout = QVBoxLayout(summary_tab)
            summary_layout.setContentsMargins(16, 16, 16, 16)
            summary_layout.setSpacing(16)

            stats_row = QHBoxLayout()
            stats_row.setSpacing(12)
            self.summary_metrics: dict[str, object] = {}
            for key, label in (
                ("kernels", "Kernels analyzed"),
                ("occupancy", "Average occupancy"),
                ("divergence", "Divergence sites"),
                ("spills", "Spill operations"),
            ):
                card, value = self._make_stat_card(label)
                self.summary_metrics[key] = value
                stats_row.addWidget(card)

            self.summary_view = QTextEdit()
            self.summary_view.setObjectName("reportPane")
            self.summary_view.setReadOnly(True)
            summary_layout.addLayout(stats_row)
            summary_layout.addWidget(self.summary_view)
            self.tabs.addTab(summary_tab, "Summary")

            self.table = QTableWidget(0, 7)
            self.table.setHorizontalHeaderLabels(
                [
                    "Kernel",
                    "Occupancy %",
                    "Limiter",
                    "Regs",
                    "Spills",
                    "Divergence",
                    "Missing Sync",
                ]
            )
            self.table.setAlternatingRowColors(True)
            self.table.setShowGrid(False)
            self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
            self.table.verticalHeader().setVisible(False)
            self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            self.tabs.addTab(self.table, "Metrics")

            self.json_view = QTextEdit()
            self.json_view.setObjectName("jsonPane")
            self.json_view.setReadOnly(True)
            self.tabs.addTab(self.json_view, "JSON")

            self.guidance_view = QTextEdit()
            self.guidance_view.setObjectName("reportPane")
            self.guidance_view.setReadOnly(True)
            self.tabs.addTab(self.guidance_view, "Action Plan")

            self.diff_view = QTextEdit()
            self.diff_view.setObjectName("diffPane")
            self.diff_view.setReadOnly(True)
            self.tabs.addTab(self.diff_view, "Diff")

            self.table.itemSelectionChanged.connect(self._update_guidance_from_selection)
            for idx, label in enumerate(_METRIC_TOOLTIPS):
                header_item = self.table.horizontalHeaderItem(idx)
                if header_item is not None:
                    header_item.setToolTip(_METRIC_TOOLTIPS[label])

            layout.addWidget(hero)
            layout.addWidget(controls)
            layout.addWidget(self.tabs)

            # --- Accessibility: logical tab order ---
            # Follows reading order: PTX path -> Browse -> Arch -> Threads ->
            # Filter -> Curve checkbox -> Run -> Save JSON -> tabs
            self.setTabOrder(self.ptx_path, self.arch_combo)
            self.setTabOrder(self.arch_combo, self.threads_spin)
            self.setTabOrder(self.threads_spin, self.kernel_filter)
            self.setTabOrder(self.kernel_filter, self.curve_box)

            self.statusBar().showMessage(
                "Ready - open a PTX file to generate a styled desktop analysis report"
            )

        def _change_theme(self, theme_name: str) -> None:
            if not theme_name or not self.theme_support:
                return
            self.current_theme = theme_name
            _apply_app_theme(self.app, self.theme_support, theme_name)
            self.statusBar().showMessage(f"Theme switched to {theme_name}", 3000)

        def _refresh_summary_metrics(self, reports: list[dict]) -> None:
            if not reports:
                self.summary_metrics["kernels"].setText("0")
                self.summary_metrics["occupancy"].setText("0%")
                self.summary_metrics["divergence"].setText("0")
                self.summary_metrics["spills"].setText("0")
                return

            avg_occ = sum(report["occupancy"]["percent"] for report in reports) / len(reports)
            total_div = sum(report["divergence"]["site_count"] for report in reports)
            total_spills = sum(report["memory"]["spill_ops"] for report in reports)

            self.summary_metrics["kernels"].setText(str(len(reports)))
            self.summary_metrics["occupancy"].setText(f"{avg_occ:.1f}%")
            self.summary_metrics["divergence"].setText(str(total_div))
            self.summary_metrics["spills"].setText(str(total_spills))

        def _clear_analysis_views(self) -> None:
            self.last_reports = []
            self.summary_view.clear()
            self.json_view.clear()
            self.guidance_view.setHtml(_format_overall_summary_html([]))
            self.table.setRowCount(0)
            self._refresh_summary_metrics([])

        def _update_guidance_from_selection(self) -> None:
            selected_ranges = self.table.selectedRanges()
            if not selected_ranges or not self.last_reports:
                self.guidance_view.setHtml(_format_overall_summary_html(self.last_reports))
                return

            row = selected_ranges[0].topRow()
            if 0 <= row < len(self.last_reports):
                self.guidance_view.setHtml(_format_action_plan_html(self.last_reports[row]))

        def _pick_ptx_file(self) -> None:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select PTX file",
                "",
                "PTX files (*.ptx);;All files (*)",
            )
            if path:
                self.ptx_path.setText(path)

        def _pick_baseline(self) -> None:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select baseline PTX",
                "",
                "PTX files (*.ptx);;All files (*)",
            )
            if path:
                self.base_path.setText(path)

        def _pick_optimized(self) -> None:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select optimized PTX",
                "",
                "PTX files (*.ptx);;All files (*)",
            )
            if path:
                self.opt_path.setText(path)

        def _run_analysis(self) -> None:
            raw_path = self.ptx_path.text().strip()
            if not raw_path:
                QMessageBox.warning(self, "Missing Input", "Choose a PTX file first.")
                return

            ptx_file = Path(raw_path)
            if not ptx_file.exists() or not ptx_file.is_file():
                QMessageBox.critical(self, "Invalid File", f"PTX file not found: {ptx_file}")
                return

            arch_spec = get_arch(self.arch_combo.currentText())
            threads = int(self.threads_spin.value())
            kernel_filter = self.kernel_filter.text().strip()
            include_curve = self.curve_box.isChecked()

            try:
                kernels = self.parser.parse_file(ptx_file)
            except Exception as exc:
                QMessageBox.critical(self, "Parse Error", str(exc))
                return

            filtered = [k for k in kernels if (not kernel_filter or kernel_filter in k.name)]
            if not filtered:
                QMessageBox.information(self, "No Kernels", "No matching kernels were found.")
                self._clear_analysis_views()
                return

            reports: list[dict] = []
            self.table.setRowCount(0)

            for idx, kernel in enumerate(filtered):
                occ = self.occ.analyze(kernel, arch_spec, threads)
                div = self.div.analyze(kernel)
                mem = self.mem.analyze(kernel)
                curve = self.occ.occupancy_curve(kernel, arch_spec) if include_curve else None
                report = build_json_report(kernel, occ, div, mem, curve)
                reports.append(report)

                self.table.insertRow(idx)
                row_values = [
                    report["kernel"],
                    str(report["occupancy"]["percent"]),
                    report["occupancy"]["limiting_factor"],
                    str(report["overview"]["registers"]),
                    str(report["memory"]["spill_ops"]),
                    str(report["divergence"]["site_count"]),
                    str(report["memory"]["possible_missing_sync"]),
                ]
                for column, value in enumerate(row_values):
                    item = QTableWidgetItem(value)
                    item.setToolTip(_METRIC_TOOLTIPS[self.table.horizontalHeaderItem(column).text()])
                    self.table.setItem(idx, column, item)

            self.last_reports = reports
            self._refresh_summary_metrics(reports)
            self.summary_view.setHtml(_format_overall_summary_html(reports))
            self.json_view.setPlainText(json.dumps(reports, indent=2))
            self.guidance_view.setHtml(_format_action_plan_html(reports[0]))
            if self.table.rowCount() > 0:
                self.table.selectRow(0)
            self.statusBar().showMessage(f"Analyzed {len(reports)} kernel(s)")
            self.tabs.setCurrentIndex(0)

        def _run_diff(self) -> None:
            base = Path(self.base_path.text().strip())
            opt = Path(self.opt_path.text().strip())
            if not base.exists() or not base.is_file():
                QMessageBox.warning(self, "Missing Baseline", "Select a valid baseline PTX file.")
                return
            if not opt.exists() or not opt.is_file():
                QMessageBox.warning(self, "Missing Optimized", "Select a valid optimized PTX file.")
                return

            arch_spec = get_arch(self.arch_combo.currentText())
            threads = int(self.threads_spin.value())

            try:
                base_kernels = {k.name: k for k in self.parser.parse_file(base)}
                opt_kernels = {k.name: k for k in self.parser.parse_file(opt)}
            except Exception as exc:
                QMessageBox.critical(self, "Diff Error", str(exc))
                return

            common = sorted(set(base_kernels) & set(opt_kernels))
            if not common:
                self.diff_view.setHtml(
                    "<h2>No matching kernel names</h2>"
                    "<p>The baseline and optimized PTX files do not expose overlapping kernel names.</p>"
                )
                self.tabs.setCurrentIndex(3)
                return

            entries: list[dict] = []
            for name in common:
                bk = base_kernels[name]
                ok = opt_kernels[name]

                b_occ = self.occ.analyze(bk, arch_spec, threads)
                o_occ = self.occ.analyze(ok, arch_spec, threads)
                b_div = self.div.analyze(bk)
                o_div = self.div.analyze(ok)
                b_mem = self.mem.analyze(bk)
                o_mem = self.mem.analyze(ok)

                d_occ = o_occ.occupancy - b_occ.occupancy
                d_regs = ok.registers.physical_regs - bk.registers.physical_regs
                d_spill = sum(w.count for w in o_mem.spill_warnings) - sum(w.count for w in b_mem.spill_warnings)
                d_div = len(o_div.sites) - len(b_div.sites)

                is_regression = (d_occ < -0.05) or (d_regs > 8) or (d_spill > 0) or (d_div > 0)
                is_improvement = (d_occ > 0.05) or (d_regs < 0 and d_spill <= 0)
                verdict = "REGRESSION" if is_regression else ("IMPROVED" if is_improvement else "NEUTRAL")

                entries.append(
                    {
                        "kernel": name,
                        "occupancy_delta": f"{d_occ:+.1%}",
                        "register_delta": f"{d_regs:+d}",
                        "spill_delta": f"{d_spill:+d}",
                        "divergence_delta": f"{d_div:+d}",
                        "verdict": verdict,
                    }
                )

            self.last_diff_entries = entries
            self.diff_view.setHtml(_format_diff_html(entries, base.name, opt.name, arch_spec.sm))
            self.statusBar().showMessage(f"Diff completed for {len(common)} kernel(s)")
            self.tabs.setCurrentIndex(3)

        def _save_json(self) -> None:
            if not self.last_reports:
                QMessageBox.information(self, "No Data", "Run analysis first to export JSON.")
                return
            path, _ = QFileDialog.getSaveFileName(self, "Save analysis JSON", "report.json", "JSON files (*.json)")
            if not path:
                return
            out = Path(path)
            if out.parent and not out.parent.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(self.last_reports, indent=2), encoding="utf-8")
            self.statusBar().showMessage(f"Saved JSON report: {out}")

    app = QApplication(sys.argv)
    app.setApplicationName("cuda-sage")
    _apply_app_theme(
        app,
        theme_support,
        "dark_teal.xml" if theme_support and "dark_teal.xml" in theme_support["themes"] else "built-in",
    )
    window = MainWindow(app, theme_support)
    window.show()

    capture_dir_raw = os.getenv("CUDA_SAGE_CAPTURE_DIR", "").strip()
    if capture_dir_raw:
        capture_dir = Path(capture_dir_raw)

        def _flush() -> None:
            """Force a complete layout + paint cycle for offscreen rendering."""
            for _ in range(6):
                app.processEvents()

        def _grab(filename: str) -> None:
            _flush()
            window.grab().save(str(capture_dir / filename))

        def _capture() -> None:
            capture_dir.mkdir(parents=True, exist_ok=True)
            repo_root = Path(__file__).resolve().parents[2]
            fixtures = repo_root / "tests" / "fixtures"

            # Force layout to calculate before any grab: resize triggers a full
            # geometry pass that QFormLayout needs to distribute widths correctly.
            window.resize(1280, 860)
            window.show()
            window.raise_()
            _flush()

            # Drive the existing UI handlers so screenshots reflect real app behavior.
            window.ptx_path.setText(str(fixtures / "vecadd.ptx"))
            window.arch_combo.setCurrentText("sm_80")
            window.threads_spin.setValue(256)
            window.kernel_filter.setText("")
            window.curve_box.setChecked(True)
            window._run_analysis()
            _flush()

            window.tabs.setCurrentIndex(0)
            _grab("gui-summary.png")

            window.tabs.setCurrentIndex(1)
            _grab("gui-metrics.png")

            window.tabs.setCurrentIndex(2)
            _grab("gui-json.png")

            window.tabs.setCurrentIndex(3)
            _grab("gui-action-plan.png")

            window.base_path.setText(str(fixtures / "vecadd.ptx"))
            window.opt_path.setText(str(fixtures / "vecadd.ptx"))
            window._run_diff()
            _flush()
            window.tabs.setCurrentIndex(4)
            _grab("gui-diff.png")

            app.quit()

        QTimer.singleShot(500, _capture)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()