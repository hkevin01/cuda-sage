"""PyQt6 desktop GUI for cuda-sage.

This module provides a native desktop interface for users who prefer not to use
CLI commands directly.
"""

from __future__ import annotations

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


def _require_pyqt6() -> dict[str, object]:
    try:
        from PyQt6.QtCore import Qt, QTimer
        from PyQt6.QtWidgets import (
            QApplication,
            QCheckBox,
            QComboBox,
            QFileDialog,
            QGridLayout,
            QGroupBox,
            QHBoxLayout,
            QHeaderView,
            QLabel,
            QLineEdit,
            QMainWindow,
            QMessageBox,
            QPushButton,
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
        "QCheckBox": QCheckBox,
        "QComboBox": QComboBox,
        "QFileDialog": QFileDialog,
        "QGridLayout": QGridLayout,
        "QGroupBox": QGroupBox,
        "QHBoxLayout": QHBoxLayout,
        "QHeaderView": QHeaderView,
        "QLabel": QLabel,
        "QLineEdit": QLineEdit,
        "QMainWindow": QMainWindow,
        "QMessageBox": QMessageBox,
        "QPushButton": QPushButton,
        "QSpinBox": QSpinBox,
        "QTableWidget": QTableWidget,
        "QTableWidgetItem": QTableWidgetItem,
        "QTabWidget": QTabWidget,
        "QTextEdit": QTextEdit,
        "QVBoxLayout": QVBoxLayout,
        "QWidget": QWidget,
    }


def _format_overview(report: dict) -> str:
    lines = []
    lines.append(f"Kernel: {report['kernel']} ({report['sm_target']})")
    lines.append(
        f"Occupancy: {report['occupancy']['percent']}% | "
        f"Limit: {report['occupancy']['limiting_factor']}"
    )
    lines.append(
        f"Registers: {report['overview']['registers']} | "
        f"Shared: {report['overview']['shared_mem_bytes']} bytes"
    )
    lines.append(
        f"Divergence sites: {report['divergence']['site_count']} "
        f"(high: {report['divergence']['high_severity_count']})"
    )
    lines.append(
        f"Spill ops: {report['memory']['spill_ops']} | "
        f"Mem-bound likely: {report['memory']['memory_bound_likely']}"
    )
    return "\n".join(lines)


def main() -> None:
    qt = _require_pyqt6()

    Qt = qt["Qt"]
    QTimer = qt["QTimer"]
    QApplication = qt["QApplication"]
    QCheckBox = qt["QCheckBox"]
    QComboBox = qt["QComboBox"]
    QFileDialog = qt["QFileDialog"]
    QGridLayout = qt["QGridLayout"]
    QGroupBox = qt["QGroupBox"]
    QHBoxLayout = qt["QHBoxLayout"]
    QHeaderView = qt["QHeaderView"]
    QLabel = qt["QLabel"]
    QLineEdit = qt["QLineEdit"]
    QMainWindow = qt["QMainWindow"]
    QMessageBox = qt["QMessageBox"]
    QPushButton = qt["QPushButton"]
    QSpinBox = qt["QSpinBox"]
    QTableWidget = qt["QTableWidget"]
    QTableWidgetItem = qt["QTableWidgetItem"]
    QTabWidget = qt["QTabWidget"]
    QTextEdit = qt["QTextEdit"]
    QVBoxLayout = qt["QVBoxLayout"]
    QWidget = qt["QWidget"]

    class MainWindow(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("cuda-sage Desktop")
            self.resize(1120, 760)

            self.parser = PTXParser()
            self.occ = OccupancyAnalyzer()
            self.div = DivergenceAnalyzer()
            self.mem = MemoryAnalyzer()
            self.last_reports: list[dict] = []

            self._build_ui()

        def _build_ui(self) -> None:
            root = QWidget()
            self.setCentralWidget(root)
            layout = QVBoxLayout(root)

            analyze_group = QGroupBox("Analyze PTX")
            analyze_layout = QGridLayout(analyze_group)

            self.ptx_path = QLineEdit()
            pick_ptx_btn = QPushButton("Browse...")
            pick_ptx_btn.clicked.connect(self._pick_ptx_file)

            self.arch_combo = QComboBox()
            for sm in sorted(ARCHITECTURES.keys()):
                self.arch_combo.addItem(sm)
            self.arch_combo.setCurrentText("sm_80")

            self.threads_spin = QSpinBox()
            self.threads_spin.setRange(1, 1024)
            self.threads_spin.setSingleStep(32)
            self.threads_spin.setValue(256)

            self.kernel_filter = QLineEdit()
            self.kernel_filter.setPlaceholderText("Optional kernel name substring")

            self.curve_box = QCheckBox("Include occupancy curve")

            run_btn = QPushButton("Run Analysis")
            run_btn.clicked.connect(self._run_analysis)

            save_json_btn = QPushButton("Save JSON")
            save_json_btn.clicked.connect(self._save_json)

            analyze_layout.addWidget(QLabel("PTX file"), 0, 0)
            analyze_layout.addWidget(self.ptx_path, 0, 1)
            analyze_layout.addWidget(pick_ptx_btn, 0, 2)
            analyze_layout.addWidget(QLabel("Architecture"), 1, 0)
            analyze_layout.addWidget(self.arch_combo, 1, 1)
            analyze_layout.addWidget(QLabel("Threads/block"), 1, 2)
            analyze_layout.addWidget(self.threads_spin, 1, 3)
            analyze_layout.addWidget(QLabel("Kernel filter"), 2, 0)
            analyze_layout.addWidget(self.kernel_filter, 2, 1, 1, 3)
            analyze_layout.addWidget(self.curve_box, 3, 0, 1, 2)

            action_row = QHBoxLayout()
            action_row.addWidget(run_btn)
            action_row.addWidget(save_json_btn)
            action_row.addStretch(1)
            analyze_layout.addLayout(action_row, 3, 2, 1, 2)

            diff_group = QGroupBox("Diff PTX")
            diff_layout = QGridLayout(diff_group)

            self.base_path = QLineEdit()
            base_btn = QPushButton("Browse baseline...")
            base_btn.clicked.connect(self._pick_baseline)

            self.opt_path = QLineEdit()
            opt_btn = QPushButton("Browse optimized...")
            opt_btn.clicked.connect(self._pick_optimized)

            run_diff_btn = QPushButton("Run Diff")
            run_diff_btn.clicked.connect(self._run_diff)

            diff_layout.addWidget(QLabel("Baseline PTX"), 0, 0)
            diff_layout.addWidget(self.base_path, 0, 1)
            diff_layout.addWidget(base_btn, 0, 2)
            diff_layout.addWidget(QLabel("Optimized PTX"), 1, 0)
            diff_layout.addWidget(self.opt_path, 1, 1)
            diff_layout.addWidget(opt_btn, 1, 2)
            diff_layout.addWidget(run_diff_btn, 2, 2)

            self.tabs = QTabWidget()

            self.summary_view = QTextEdit()
            self.summary_view.setReadOnly(True)
            self.tabs.addTab(self.summary_view, "Summary")

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
            self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            self.tabs.addTab(self.table, "Metrics")

            self.json_view = QTextEdit()
            self.json_view.setReadOnly(True)
            self.tabs.addTab(self.json_view, "JSON")

            self.diff_view = QTextEdit()
            self.diff_view.setReadOnly(True)
            self.tabs.addTab(self.diff_view, "Diff")

            layout.addWidget(analyze_group)
            layout.addWidget(diff_group)
            layout.addWidget(self.tabs)

            self.statusBar().showMessage("Ready")

        def _pick_ptx_file(self) -> None:
            path, _ = QFileDialog.getOpenFileName(self, "Select PTX file", "", "PTX files (*.ptx);;All files (*)")
            if path:
                self.ptx_path.setText(path)

        def _pick_baseline(self) -> None:
            path, _ = QFileDialog.getOpenFileName(self, "Select baseline PTX", "", "PTX files (*.ptx);;All files (*)")
            if path:
                self.base_path.setText(path)

        def _pick_optimized(self) -> None:
            path, _ = QFileDialog.getOpenFileName(self, "Select optimized PTX", "", "PTX files (*.ptx);;All files (*)")
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
                self.last_reports = []
                self.summary_view.clear()
                self.json_view.clear()
                self.table.setRowCount(0)
                return

            reports: list[dict] = []
            summary_blocks: list[str] = []
            self.table.setRowCount(0)

            for idx, kernel in enumerate(filtered):
                occ = self.occ.analyze(kernel, arch_spec, threads)
                div = self.div.analyze(kernel)
                mem = self.mem.analyze(kernel)
                curve = self.occ.occupancy_curve(kernel, arch_spec) if include_curve else None
                report = build_json_report(kernel, occ, div, mem, curve)
                reports.append(report)
                summary_blocks.append(_format_overview(report))

                self.table.insertRow(idx)
                self.table.setItem(idx, 0, QTableWidgetItem(report["kernel"]))
                self.table.setItem(idx, 1, QTableWidgetItem(str(report["occupancy"]["percent"])))
                self.table.setItem(idx, 2, QTableWidgetItem(report["occupancy"]["limiting_factor"]))
                self.table.setItem(idx, 3, QTableWidgetItem(str(report["overview"]["registers"])))
                self.table.setItem(idx, 4, QTableWidgetItem(str(report["memory"]["spill_ops"])))
                self.table.setItem(idx, 5, QTableWidgetItem(str(report["divergence"]["site_count"])))
                self.table.setItem(idx, 6, QTableWidgetItem(str(report["memory"]["possible_missing_sync"])))

            self.last_reports = reports
            self.summary_view.setPlainText("\n\n".join(summary_blocks))
            self.json_view.setPlainText(json.dumps(reports, indent=2))
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
                self.diff_view.setPlainText("No matching kernel names between baseline and optimized PTX files.")
                self.tabs.setCurrentIndex(3)
                return

            lines = [f"Diff: {base.name} -> {opt.name} ({arch_spec.sm})", ""]
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

                lines.append(f"Kernel: {name}")
                lines.append(f"  Occupancy delta: {d_occ:+.1%}")
                lines.append(f"  Register delta: {d_regs:+d}")
                lines.append(f"  Spill delta: {d_spill:+d}")
                lines.append(f"  Divergence-site delta: {d_div:+d}")
                lines.append(f"  Verdict: {verdict}")
                lines.append("")

            self.diff_view.setPlainText("\n".join(lines).rstrip())
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
    window = MainWindow()
    window.show()

    capture_dir_raw = os.getenv("CUDA_SAGE_CAPTURE_DIR", "").strip()
    if capture_dir_raw:
        capture_dir = Path(capture_dir_raw)

        def _capture() -> None:
            capture_dir.mkdir(parents=True, exist_ok=True)
            repo_root = Path(__file__).resolve().parents[2]
            fixtures = repo_root / "tests" / "fixtures"

            # Drive the existing UI handlers so screenshots reflect real app behavior.
            window.ptx_path.setText(str(fixtures / "vecadd.ptx"))
            window.arch_combo.setCurrentText("sm_80")
            window.threads_spin.setValue(256)
            window.kernel_filter.setText("")
            window.curve_box.setChecked(True)
            window._run_analysis()

            app.processEvents()
            window.tabs.setCurrentIndex(0)
            app.processEvents()
            window.grab().save(str(capture_dir / "gui-summary.png"))

            window.tabs.setCurrentIndex(1)
            app.processEvents()
            window.grab().save(str(capture_dir / "gui-metrics.png"))

            window.tabs.setCurrentIndex(2)
            app.processEvents()
            window.grab().save(str(capture_dir / "gui-json.png"))

            window.base_path.setText(str(fixtures / "vecadd.ptx"))
            window.opt_path.setText(str(fixtures / "vecadd.ptx"))
            window._run_diff()
            app.processEvents()
            window.tabs.setCurrentIndex(3)
            app.processEvents()
            window.grab().save(str(capture_dir / "gui-diff.png"))

            app.quit()

        QTimer.singleShot(350, _capture)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()