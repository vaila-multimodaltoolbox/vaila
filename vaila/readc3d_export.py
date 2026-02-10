"""
===============================================================================
readc3d_export.py
===============================================================================
Author: Paulo R. P. Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 25 September 2024
Update Date: 03 February 2026
Version: 0.2.1

Description:
This script processes .c3d files, extracting marker data, analog data, events, and points residuals,
and saves them into CSV files. It also allows the option to save the data in Excel format.
The script leverages Dask for efficient data handling and processing, particularly useful
when working with large datasets.

Features:
- Extracts and saves marker data with time columns.
- Extracts and saves analog data with time columns, including their units.
- Extracts and saves events with their labels and times.
- Extracts and saves points residuals with time columns.
- Supports saving the data in CSV format.
- Optionally saves the data in Excel format (can be slow for large files).
- Generates an info file containing metadata about markers, analogs, and their units.
- Generates a simplified short info file with key parameters and headers.
- Handles encoding errors to avoid crashes due to unexpected characters.
- Extracts and saves force platform data including Center of Pressure (COP) from the C3D file into CSV files.
- Saves the COP data in a combined CSV file with time columns and platform indices.
- Saves a summary file with platform information.
- Saves a summary file with marker information.
- Saves a summary file with analog information.
- Saves a summary file with points residuals information.

Dependencies:
- Python 3.12.12
- ezc3d
- Pandas
- Tkinter
- Tqdm
- Numpy
- Openpyxl (optional, for saving Excel files)
- Rich (optional, for console output)

Contact:
- Paulo Roberto Pereira Santiago
- paulosantiago@usp.br

Version history:
- v0.2.0 (28 February 2025): Added support for saving COP data in a combined CSV file.

Usage:
- Run the script, select the input directory containing .c3d files, and specify an output directory.
- Choose whether to save the files in Excel format.
- The script will process each .c3d file in the input directory and save the results in the specified output directory.

Example:
Windows:
$ python readc3d_export.py

Linux:
$ python3 readc3d_export.py

macOS:
$ python3 readc3d_export.py

Notes:
- Ensure that all necessary libraries are installed.
- This script is designed to handle large datasets efficiently, but saving to Excel format may take significant time depending on the dataset size.
- The calculation and export of COP data were removed from this script.
- The COP processing will be performed later in the cop_calculate.py script.

License:
- This script is licensed under the GNU General Public License v3.0. GPLv3.
- If you use this script, please cite the following paper:

Citation:
@misc{vaila2024,
  title={vailá - Versatile Anarcho Integrated Liberation Ánalysis in Multimodal Toolbox},
  author={Paulo Roberto Pereira Santiago and Guilherme Manna Cesar and Ligia Yumi Mochida and Juan Aceros and others},
  year={2024},
  eprint={2410.07238},
  archivePrefix={arXiv},
  primaryClass={cs.HC},
  url={https://arxiv.org/abs/2410.07238}
}
"""

import json
import math
import pathlib
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import Tk, filedialog, messagebox, scrolledtext, ttk

import ezc3d
import numpy as np
import pandas as pd
from ezc3d import c3d
from rich import print
from tqdm import tqdm


# #region agent log
def _debug_log(location, message, data, hypothesis_id="H1"):
    try:
        with open(r"c:\Users\paulo\Preto\vaila\.cursor\debug.log", "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "location": location,
                        "message": message,
                        "data": data,
                        "hypothesisId": hypothesis_id,
                        "timestamp": datetime.now().isoformat(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    except Exception:
        pass


# #endregion


# === ADVANCED C3D REPORT GENERATOR ===
class C3DReportGenerator:
    """
    Classe para gerar relatórios detalhados (Inspeção Profunda) de arquivos C3D.
    Autor: Paulo R. P. Santiago (vailá Toolbox)
    """

    def __init__(self, c3d_path):
        self.path = Path(c3d_path)
        self.filename = self.path.name
        self.creation_date = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

        # Carregar C3D (fallback sem extract_forceplat_data para arquivos que falham com force platform)
        try:
            self.c3d = ezc3d.c3d(str(self.path), extract_forceplat_data=True)
        except Exception:
            try:
                self.c3d = ezc3d.c3d(str(self.path))
            except Exception as e:
                raise ValueError(f"Erro ao ler o C3D com ezc3d: {e}")

        self.header = self.c3d["header"]
        self.parameters = self.c3d["parameters"]
        self.data = self.c3d["data"]
        # #region agent log
        _debug_log(
            "C3DReportGenerator.__init__",
            "header keys after load",
            list(self.header.keys()) if isinstance(self.header, dict) else str(type(self.header)),
            "H1",
        )
        _debug_log(
            "C3DReportGenerator.__init__",
            "has version?",
            "version" in self.header if isinstance(self.header, dict) else "N/A",
            "H1",
        )
        # #endregion

    def _get(self, obj, key, default=None):
        """Get value from dict or object (ezc3d Header returns objects, not dicts)."""
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def generate_txt_report(self, output_path=None):
        if not output_path:
            output_path = self.path.with_suffix(".txt")
        # #region agent log
        _debug_log(
            "generate_txt_report",
            "entry",
            {
                "output_path": str(output_path),
                "header_keys": list(self.header.keys())
                if isinstance(self.header, dict)
                else "not_dict",
            },
            "H2",
        )
        # #endregion
        lines = []
        lines.append("=" * 80)
        lines.append("RELATÓRIO DE INSPEÇÃO C3D - vailá Multimodal Toolbox")
        lines.append(f"Arquivo: {self.filename}")
        lines.append(f"Data da Inspeção: {self.creation_date}")
        lines.append("=" * 80)
        lines.append("")

        # 1. Cabeçalho
        lines.append("1. CABEÇALHO (HEADER)")
        lines.append("-" * 40)
        pts = self._get(self.header, "points")
        ana = self._get(self.header, "analogs")
        fr = self._get(pts, "frame_rate") or 1
        first = self._get(pts, "first_frame") or 0
        last = self._get(pts, "last_frame") or 0
        duration = (last - first + 1) / fr if fr else 0
        version = self._get(self.header, "version") or self._get(self.header, "Version") or "N/A"

        lines.append(f"Versão C3D: {version}")
        lines.append(f"Taxa de Pontos (Video): {fr} Hz")
        lines.append(f"Taxa de Analógicos:    {self._get(ana, 'frame_rate') or 0} Hz")
        lines.append(f"Total Frames (Vídeo):  {last - first + 1}")
        lines.append(f"Duração Estimada:      {duration:.2f} s")
        lines.append("")

        # 2. Eventos
        lines.append("2. EVENTOS")
        lines.append("-" * 40)
        events = self._get_events_list()
        if events:
            for ev in events:
                lines.append(f"[{ev['time']:.3f}s] {ev['label']} ({ev['context']})")
        else:
            lines.append("Nenhum evento registrado.")
        lines.append("")

        # 3. Estatísticas de Marcadores (Health Check)
        lines.append("3. SAÚDE DOS MARCADORES (Gaps/Oclusões)")
        lines.append("-" * 40)
        marker_stats = self._calculate_marker_health()
        for m in marker_stats:
            lines.append(
                f"{m['name']:<20} | NaNs: {m['nans']:>5} ({m['pct']:.1f}%) | Max Gap: {m['max_gap_frames']} f"
            )
        lines.append("")

        # 4. Parâmetros Completos
        lines.append("4. ÁRVORE DE PARÂMETROS COMPLETA")
        lines.append("-" * 40)
        for group_name, group in self.parameters.items():
            lines.append(f"\n[GRUPO: {group_name}]")
            lines.append(f"  Descrição: {group.get('description', 'N/A')}")

            for param_name, param in group.items():
                if param_name == "__METADATA__":
                    continue
                val_str = str(param["value"])
                # Truncar valores muito longos no TXT
                if len(val_str) > 100:
                    val_str = val_str[:97] + "..."

                lines.append(f"  > {param_name}: {val_str}")
                lines.append(f"    Desc: {param.get('description', '')}")
                lines.append(f"    Type: {param.get('type', '')}")

        with open(output_path, "w", encoding="utf-8", errors="replace") as f:
            f.write("\n".join(lines))

        return output_path

    def generate_html_report(self, output_path=None):
        if not output_path:
            output_path = self.path.with_suffix(".html")
        # #region agent log
        _debug_log(
            "generate_html_report",
            "entry",
            {
                "output_path": str(output_path),
                "header_keys": list(self.header.keys())
                if isinstance(self.header, dict)
                else "not_dict",
            },
            "H2",
        )
        # #endregion
        # Coletar dados pré-processados
        marker_stats = self._calculate_marker_health()
        events = self._get_events_list()
        analog_info = self._get_analog_info()

        # Header values (ezc3d returns Header object, not dict)
        pts = self._get(self.header, "points")
        ana = self._get(self.header, "analogs")
        _version = self._get(self.header, "version") or self._get(self.header, "Version") or "N/A"
        _pt_fr = self._get(pts, "frame_rate") or 0
        _ana_fr = self._get(ana, "frame_rate") or 0
        _first = self._get(pts, "first_frame") or 0
        _last = self._get(pts, "last_frame") or 0
        _duration = ((_last - _first + 1) / _pt_fr) if _pt_fr else 0

        # Construção do HTML
        html_content = f"""
        <!DOCTYPE html>
        <html lang="pt-BR">
        <head>
            <meta charset="UTF-8">
            <title>Inspeção C3D: {self.filename}</title>
            <style>
                body {{ font-family: 'Segoe UI', Arial, sans-serif; background-color: #f4f6f9; color: #333; line-height: 1.6; margin: 0; padding: 20px; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
                h1 {{ border-bottom: 3px solid #3498db; color: #2c3e50; padding-bottom: 10px; }}
                h2 {{ color: #2980b9; margin-top: 30px; border-left: 5px solid #3498db; padding-left: 10px; }}
                h3 {{ color: #16a085; margin-top: 20px; }}

                table {{ width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 0.9em; }}
                th {{ background-color: #2c3e50; color: white; text-align: left; padding: 12px; }}
                td {{ border-bottom: 1px solid #ddd; padding: 8px; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #e8f6f3; }}

                .warning {{ color: #e74c3c; font-weight: bold; }}
                .good {{ color: #27ae60; font-weight: bold; }}

                details {{ background: #eee; margin-bottom: 10px; border-radius: 4px; overflow: hidden; }}
                summary {{ cursor: pointer; padding: 10px; font-weight: bold; background: #dfe6e9; }}
                summary:hover {{ background: #b2bec3; }}
                .param-content {{ padding: 15px; background: white; border: 1px solid #dfe6e9; }}

                .badge {{ background: #3498db; color: white; padding: 2px 6px; border-radius: 4px; font-size: 0.8em; }}
                .desc {{ color: #7f8c8d; font-style: italic; font-size: 0.85em; display: block; margin-top: 4px; }}

                .footer {{ margin-top: 50px; text-align: center; color: #7f8c8d; font-size: 0.8em; border-top: 1px solid #eee; padding-top: 20px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Inspeção C3D: {self.filename}</h1>
                <p><strong>Caminho:</strong> {self.path}<br>
                <strong>Gerado em:</strong> {self.creation_date} via <em>vailá Multimodal Toolbox</em></p>

                <h2>1. Visão Geral da Aquisição</h2>
                <table>
                    <tr><th>Parâmetro</th><th>Valor</th></tr>
                    <tr><td>Versão do Arquivo</td><td>{_version}</td></tr>
                    <tr><td>Frequência de Pontos (Vídeo)</td><td>{_pt_fr} Hz</td></tr>
                    <tr><td>Frequência de Analógicos</td><td>{_ana_fr} Hz</td></tr>
                    <tr><td>Primeiro Frame</td><td>{_first}</td></tr>
                    <tr><td>Último Frame</td><td>{_last}</td></tr>
                    <tr><td>Duração Total</td><td>{_duration:.2f} segundos</td></tr>
                </table>

                <h2>2. Saúde dos Marcadores (Data Quality)</h2>
                <p>Análise de consistência e gaps nas trajetórias 3D.</p>
                <table>
                    <tr>
                        <th>Marcador</th>
                        <th>Frames Válidos</th>
                        <th>NaNs (Gaps)</th>
                        <th>% Perda</th>
                        <th>Status</th>
                    </tr>
        """

        # Loop Marcadores
        for m in marker_stats:
            status_class = "warning" if m["pct"] > 5.0 else "good"
            status_text = "ALERTA" if m["pct"] > 5.0 else "OK"
            html_content += f"""
                    <tr>
                        <td>{m["name"]}</td>
                        <td>{m["valid"]}</td>
                        <td>{m["nans"]}</td>
                        <td>{m["pct"]:.2f}%</td>
                        <td class="{status_class}">{status_text}</td>
                    </tr>
            """

        html_content += """
                </table>

                <h2>3. Canais Analógicos</h2>
                <table>
                    <tr><th>Canal</th><th>Unidade</th><th>Mínimo</th><th>Máximo</th></tr>
        """

        for a in analog_info:
            html_content += f"<tr><td>{a['name']}</td><td>{a['unit']}</td><td>{a['min']:.4f}</td><td>{a['max']:.4f}</td></tr>"

        html_content += """
                </table>

                <h2>4. Eventos Temporais</h2>
        """
        if events:
            html_content += "<table><tr><th>Tempo (s)</th><th>Label</th><th>Contexto</th></tr>"
            for ev in events:
                html_content += f"<tr><td>{ev['time']:.3f}</td><td>{ev['label']}</td><td>{ev['context']}</td></tr>"
            html_content += "</table>"
        else:
            html_content += "<p><em>Nenhum evento registrado neste arquivo.</em></p>"

        # SECTION 5: PARAMETERS TREE
        html_content += """
                <h2>5. Dicionário de Parâmetros (Estrutura Completa)</h2>
                <p>Clique nos grupos abaixo para expandir os metadados técnicos.</p>
        """

        for group_name, group in self.parameters.items():
            desc = group.get("description", "")
            html_content += f"""
                <details>
                    <summary>{group_name} <span style="font-weight:normal; font-size:0.8em">({desc})</span></summary>
                    <div class="param-content">
                        <table>
                            <tr><th width="20%">Parâmetro</th><th width="50%">Valor</th><th width="30%">Detalhes</th></tr>
            """

            for param_name, param in group.items():
                if param_name == "__METADATA__":
                    continue

                val = param["value"]
                # Formatação segura para visualização
                if isinstance(val, np.ndarray):
                    val_display = f"Array {val.shape}"
                    if val.size < 10:
                        val_display = str(val)
                elif isinstance(val, list) and len(val) > 10:
                    val_display = f"List [{len(val)} items] (ver raw data)"
                else:
                    val_display = str(val)

                val_display = str(val)
                p_desc = param.get("description", "")
                p_type = param.get("type", "?")

                html_content += f"""
                            <tr>
                                <td><strong>{param_name}</strong></td>
                                <td style="font-family:monospace; color:#d63031;">{val_display}</td>
                                <td>
                                    <span class="badge">{p_type}</span>
                                    <span class="desc">{p_desc}</span>
                                </td>
                            </tr>
                """

            html_content += """
                        </table>
                    </div>
                </details>
            """

        html_content += """
                <div class="footer">
                    Relatório gerado automaticamente por <strong>vailá Multimodal Toolbox</strong>.<br>
                    Professor Paulo R. P. Santiago | USP
                </div>
            </div>
        </body>
        </html>
        """

        with open(output_path, "w", encoding="utf-8", errors="replace") as f:
            f.write(html_content)

        return output_path

    def _label_to_str(self, label):
        """Normalize label to string (ezc3d may return list of lists, e.g. [['LASI'], ...])."""
        if isinstance(label, (list, np.ndarray)) and len(label) > 0:
            return str(label[0]).strip()
        return str(label).strip()

    def _calculate_marker_health(self):
        """Analisa a qualidade do sinal dos marcadores (Pontos)."""
        stats = []
        if "POINT" not in self.parameters:
            return stats

        labels = self.parameters["POINT"]["LABELS"]["value"]
        # Data shape: (3, n_markers, n_frames)
        points_data = self.data["points"]

        total_frames = points_data.shape[2]

        for i, label in enumerate(labels):
            # Verificar NaNs na coordenada X (se X é nan, o ponto todo é inválido geralmente)
            # points_data[0, i, :] pega todos os frames da coord X do marcador i
            marker_slice = points_data[0:3, i, :]

            # Um frame é NaN se qualquer eixo (x,y,z) for NaN
            is_nan_frame = np.any(np.isnan(marker_slice), axis=0)
            nan_count = np.sum(is_nan_frame)
            pct = (nan_count / total_frames) * 100 if total_frames > 0 else 0

            stats.append(
                {
                    "name": self._label_to_str(label),
                    "nans": int(nan_count),
                    "valid": int(total_frames - nan_count),
                    "pct": pct,
                    "max_gap_frames": self._max_consecutive_true(is_nan_frame),
                }
            )
        return stats

    def _get_analog_info(self):
        """Retorna resumo dos analógicos."""
        info = []
        if "ANALOG" not in self.parameters:
            return info

        labels = self.parameters["ANALOG"]["LABELS"]["value"]
        units = self.parameters["ANALOG"].get("UNITS", {}).get("value", [])
        data = self.data["analogs"]

        if len(units) < len(labels):
            units = list(units) + [""] * (len(labels) - len(units))

        for i, label in enumerate(labels):
            channel_data = data[0, i, :]
            u = units[i] if i < len(units) else ""
            unit_str = (
                u
                if isinstance(u, str)
                else (str(u[0]) if isinstance(u, (list, np.ndarray)) and len(u) else str(u))
            )
            info.append(
                {
                    "name": self._label_to_str(label),
                    "unit": unit_str,
                    "min": np.nanmin(channel_data) if channel_data.size > 0 else 0,
                    "max": np.nanmax(channel_data) if channel_data.size > 0 else 0,
                }
            )
        return info

    def _get_events_list(self):
        """Normaliza a extração de eventos."""
        events_list = []
        if "EVENT" not in self.parameters:
            return events_list

        ev = self.parameters["EVENT"]
        times = ev.get("TIMES", {}).get("value", [])
        labels = ev.get("LABELS", {}).get("value", [])
        contexts = ev.get("CONTEXTS", {}).get("value", [])

        # ezc3d às vezes retorna shape (2, N) para tempos, a linha 1 são os tempos reais
        if isinstance(times, np.ndarray) and len(times.shape) > 1:
            times = times[1, :]

        # Normalizar para lista se for escalar ou array
        if not hasattr(times, "__iter__"):
            times = [times]
        if not hasattr(labels, "__iter__"):
            labels = [labels]
        if not hasattr(contexts, "__iter__"):
            contexts = [contexts]

        count = len(times)
        for i in range(count):
            t = times[i] if i < len(times) else 0.0
            l = labels[i] if i < len(labels) else ""
            c = contexts[i] if i < len(contexts) else ""
            events_list.append({"time": t, "label": l, "context": c})

        # Ordenar por tempo
        events_list.sort(key=lambda x: x["time"])
        return events_list

    def _max_consecutive_true(self, condition):
        """Helper para calcular o maior buraco (gap) de frames."""
        # Lógica rápida com numpy para achar gaps consecutivos
        padded = np.concatenate(([False], condition, [False]))
        diff = np.diff(padded.astype(int))
        starts = np.flatnonzero(diff == 1)
        ends = np.flatnonzero(diff == -1)
        if len(starts) == 0:
            return 0
        return np.max(ends - starts)


# === C3D INSPECTION TOOL (DIDACTIC VERSION) ===
class DidacticC3DInspector:
    def __init__(self, root, file_path):
        self.root = root
        self.file_path = file_path
        self.filename = Path(file_path).name

        self.root.title(f"vailá C3D Inspector - {self.filename}")
        self.root.geometry("1000x700")

        # Console Log
        print(f"Opening C3D Inspector for: {self.filename}")
        print(f"Path: {file_path}")

        # Load C3D data
        try:
            self.c3d_data = ezc3d.c3d(file_path)
            self.header = self.c3d_data["header"]
            self.params = self.c3d_data["parameters"]
            self.data = self.c3d_data["data"]
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load C3D file:\n{str(e)}")
            self.root.destroy()
            return

        self._build_ui()

    def _build_ui(self):
        # Main Style
        style = ttk.Style()
        style.configure("Bold.TLabel", font=("Arial", 10, "bold"))
        style.configure("Title.TLabel", font=("Arial", 12, "bold"), foreground="#2c3e50")

        # Notebook for Tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- TAB 1: OVERVIEW ---
        self.tab_overview = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_overview, text="Overview")
        self._build_overview_tab()

        # --- TAB 2: PARAMETERS EXPLORER (Didactic) ---
        self.tab_params = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_params, text="Parameters Map")
        self._build_parameters_tab()

        # --- TAB 3: DATA & EVENTS ---
        self.tab_data = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_data, text="Data & Events")
        self._build_data_tab()

        # Bottom Bar
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(btn_frame, text="Close", command=self.root.destroy).pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="Save HTML Report", command=self.save_html_report).pack(
            side=tk.RIGHT, padx=5
        )
        ttk.Button(btn_frame, text="Save TXT Report", command=self.save_txt_report).pack(
            side=tk.RIGHT, padx=5
        )

        # Status
        lbl_status = ttk.Label(
            btn_frame,
            text="Didactic Inspector: Explore the tabs to understand the file structure.",
            font=("Arial", 9, "italic"),
        )
        lbl_status.pack(side=tk.LEFT)

    def save_txt_report(self):
        """Uses the advanced generator."""
        report_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt")],
            initialfile=f"inspection_{self.filename}.txt",
            title="Save Advanced Inspection Report",
        )
        if not report_path:
            return

        try:
            # Chama o gerador novo
            generator = C3DReportGenerator(self.file_path)
            generator.generate_txt_report(report_path)  # Passa o caminho escolhido
            messagebox.showinfo("Success", f"Advanced report saved to:\n{report_path}")
            print(f"Advanced TXT Report saved: {report_path}")
        except Exception as e:
            # #region agent log
            _debug_log(
                "save_txt_report",
                "exception",
                {"type": type(e).__name__, "args": getattr(e, "args", ())},
                "H4",
            )
            # #endregion
            messagebox.showerror("Error", f"Failed: {e}")

    def save_html_report(self):
        """Uses the advanced generator."""
        report_path = filedialog.asksaveasfilename(
            defaultextension=".html",
            filetypes=[("HTML Files", "*.html")],
            initialfile=f"inspection_{self.filename}.html",
            title="Save Advanced HTML Report",
        )
        if not report_path:
            return

        try:
            generator = C3DReportGenerator(self.file_path)
            generator.generate_html_report(report_path)
            messagebox.showinfo("Success", f"Advanced HTML report saved to:\n{report_path}")
            print(f"Advanced HTML Report saved: {report_path}")
        except Exception as e:
            # #region agent log
            _debug_log(
                "save_html_report",
                "exception",
                {"type": type(e).__name__, "args": getattr(e, "args", ())},
                "H4",
            )
            # #endregion
            messagebox.showerror("Error", f"Failed: {e}")

    def _build_overview_tab(self):
        frame = ttk.Frame(self.tab_overview)
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # File Info
        ttk.Label(frame, text="File Information", style="Title.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 10)
        )

        info_grid = ttk.Frame(frame)
        info_grid.grid(row=1, column=0, sticky="ew")

        self._add_info_row(info_grid, 0, "Filename:", self.filename)
        self._add_info_row(info_grid, 1, "Path:", self.file_path)

        # Structure Info
        ttk.Label(frame, text="Acquisition Settings", style="Title.TLabel").grid(
            row=2, column=0, sticky="w", pady=(20, 10)
        )

        points = self.header["points"]
        analogs = self.header["analogs"]

        settings_grid = ttk.Frame(frame)
        settings_grid.grid(row=3, column=0, sticky="ew")

        self._add_info_row(settings_grid, 0, "Point Frame Rate:", f"{points['frame_rate']} Hz")
        self._add_info_row(settings_grid, 1, "Analog Frame Rate:", f"{analogs['frame_rate']} Hz")
        self._add_info_row(settings_grid, 2, "First Frame:", f"{points['first_frame']}")
        self._add_info_row(settings_grid, 3, "Last Frame:", f"{points['last_frame']}")
        duration = (points["last_frame"] - points["first_frame"] + 1) / points["frame_rate"]
        self._add_info_row(settings_grid, 4, "Duration:", f"{duration:.2f} seconds")

        # Health Check
        ttk.Label(frame, text="Data Health Check", style="Title.TLabel").grid(
            row=4, column=0, sticky="w", pady=(20, 10)
        )

        health_grid = ttk.Frame(frame)
        health_grid.grid(row=5, column=0, sticky="ew")

        # Check NaNs
        pts = self.data["points"]
        nan_pts = np.isnan(pts).sum()
        total_pts = pts.size
        pct_pts = nan_pts / total_pts * 100 if total_pts > 0 else 0

        self._add_health_row(
            health_grid, 0, "Markers (Points)", f"{nan_pts} NaNs ({pct_pts:.2f}%)", pct_pts > 5
        )

    def _add_info_row(self, parent, row, label, value):
        ttk.Label(parent, text=label, style="Bold.TLabel").grid(
            row=row, column=0, sticky="w", padx=5, pady=2
        )
        ttk.Label(parent, text=str(value)).grid(row=row, column=1, sticky="w", padx=5, pady=2)

    def _add_health_row(self, parent, row, label, status_text, is_warning):
        ttk.Label(parent, text=label, style="Bold.TLabel").grid(
            row=row, column=0, sticky="w", padx=5, pady=2
        )
        color = "red" if is_warning else "green"
        lbl = tk.Label(parent, text=status_text, fg=color)
        lbl.grid(row=row, column=1, sticky="w", padx=5, pady=2)

    def _build_parameters_tab(self):
        container = ttk.Frame(self.tab_params)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Description
        ttk.Label(
            container,
            text="Double-click groups (e.g., POINT) to expand. Select parameters to view details.",
        ).pack(side=tk.TOP, fill=tk.X, pady=(0, 5))

        # Treeview
        columns = ("desc", "value")
        self.tree = ttk.Treeview(container, columns=columns, show="tree headings")
        self.tree.heading("#0", text="Parameter / Group")
        self.tree.heading("desc", text="Description")
        self.tree.heading("value", text="Value (First few items)")
        self.tree.column("#0", width=200)
        self.tree.column("desc", width=300)
        self.tree.column("value", width=400)

        # Scrollbar
        sb = ttk.Scrollbar(container, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        # Populate Tree
        self._populate_tree()

    def _populate_tree(self):
        # Common descriptions for educational purposes
        descriptions = {
            "POINT": "Contains information about 3D trajectory markers.",
            "ANALOG": "Contains data from analog devices (Force Plates, EMGs).",
            "EVENT": "Time events marked in the trial (e.g., Heel Strike).",
            "TRIAL": "General trial information.",
            "SUBJECT": "Information about the subject (Name, Height, etc.).",
        }

        for group_name in self.params:
            group = self.params[group_name]
            node_id = self.tree.insert(
                "",
                tk.END,
                text=group_name,
                values=(descriptions.get(group_name, "User defined group"), ""),
            )

            # Insert parameters inside group
            for param_name in group:
                if param_name == "__METADATA__":
                    continue
                param = group[param_name]
                desc = param.get("description", "")
                val = param.get("value", [])

                # Format value for display
                display_val = str(val)
                if isinstance(val, list) and len(val) > 5:
                    display_val = f"{val[:5]}... ({len(val)} items)"
                elif isinstance(val, np.ndarray):
                    display_val = f"Array shape {val.shape}"

                self.tree.insert(node_id, tk.END, text=param_name, values=(desc, display_val))

    def _build_data_tab(self):
        frame = ttk.Frame(self.tab_data)
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Events Section
        ttk.Label(frame, text="Timeline Events", style="Title.TLabel").pack(
            anchor="w", pady=(0, 10)
        )

        event_tree = ttk.Treeview(
            frame, columns=("time", "label", "context"), show="headings", height=8
        )
        event_tree.heading("time", text="Time (s)")
        event_tree.heading("label", text="Label")
        event_tree.heading("context", text="Context")
        event_tree.pack(fill=tk.X)

        # Populate events
        if "EVENT" in self.params:
            ev_group = self.params["EVENT"]
            times = ev_group.get("TIMES", {}).get("value", [])
            labels = ev_group.get("LABELS", {}).get("value", [])
            contexts = ev_group.get("CONTEXTS", {}).get("value", [])

            if len(times) > 0 and len(times.shape) > 1:
                real_times = times[1, :]
                for i in range(len(real_times)):
                    t = real_times[i]
                    l = labels[i] if i < len(labels) else ""
                    c = contexts[i] if i < len(contexts) else ""
                    event_tree.insert("", tk.END, values=(f"{t:.3f}", l, c))
            elif len(times) > 0:  # single dim
                for i in range(len(times)):
                    t = times[i]
                    l = labels[i] if i < len(labels) else ""
                    c = contexts[i] if i < len(contexts) else ""
                    event_tree.insert("", tk.END, values=(f"{t:.3f}", l, c))

        # Labels Lists
        ttk.Label(frame, text="Markers List", style="Title.TLabel").pack(anchor="w", pady=(20, 10))

        txt_markers = scrolledtext.ScrolledText(frame, height=5)
        txt_markers.pack(fill=tk.X)

        if "POINT" in self.params:
            lbls = self.params["POINT"].get("LABELS", {}).get("value", [])
            if lbls:
                if isinstance(lbls[0], list):
                    lbls = lbls[0]
                txt_markers.insert(tk.END, ", ".join(lbls))

        txt_markers.configure(state="disabled")


inspect_window_ref = None


def inspect_c3d_gui(parent_root=None):
    """Launcher for the Didactic Inspector."""
    global inspect_window_ref

    if parent_root is None:
        root = tk.Tk()
        root.withdraw()
    else:
        root = parent_root

    file_path = filedialog.askopenfilename(
        title="Select C3D File to Inspect", filetypes=[("C3D Files", "*.c3d")]
    )

    if not file_path:
        return

    # Use Toplevel to keep main app alive
    win = tk.Toplevel(root)
    # Store ref to prevent garbage collection issues if any
    inspect_window_ref = DidacticC3DInspector(win, file_path)

    if parent_root is None:
        root.mainloop()


# Removed legacy text-based generator

# =================================


def get_time_precision(freq):
    """
    Calculate the number of decimal places needed for time formatting based on sampling frequency.

    For frequencies <= 1000 Hz: uses 3 decimal places (0.001s precision)
    For frequencies > 1000 Hz: calculates decimal places needed to represent the sampling interval

    Args:
        freq: Sampling frequency in Hz

    Returns:
        Number of decimal places needed for time formatting
    """
    if freq <= 1000:
        return 3
    else:
        # Calculate the sampling interval
        interval = 1.0 / freq
        # Calculate number of decimal places needed
        # Use ceil of -log10(interval) to ensure we have enough precision
        # For example: 2000 Hz -> interval = 0.0005 -> -log10(0.0005) ≈ 3.3 -> ceil = 4
        # For example: 10000 Hz -> interval = 0.0001 -> -log10(0.0001) = 4 -> ceil = 4
        decimal_places = max(3, int(math.ceil(-math.log10(interval))))
        return decimal_places


def save_info_file(datac3d, file_name, output_dir):
    """
    Save all parameters and data from the C3D file into a detailed .info text file.
    """
    print(f"Saving all data to .info file for {file_name}")

    info_file_path = Path(output_dir) / f"{file_name}.info"
    # Use encoding='utf-8' and ignore errors
    with open(info_file_path, "w", encoding="utf-8", errors="ignore") as info_file:
        # Write header information
        info_file.write(f"File: {file_name}\n")
        info_file.write("--- Parameters in C3D File ---\n\n")

        # Iterate over all groups in parameters and write them to the .info file
        for group_name, group_content in datac3d["parameters"].items():
            info_file.write(f"Group: {group_name}\n")
            for param_name, param_content in group_content.items():
                info_file.write(f"  Parameter: {param_name}\n")
                info_file.write(
                    f"    Description: {param_content.get('description', 'No description')}\n"
                )
                info_file.write(f"    Value: {param_content.get('value', 'No value')}\n")
                info_file.write(f"    Type: {param_content.get('type', 'No type')}\n")
                info_file.write(
                    f"    Dimension: {param_content.get('dimension', 'No dimension')}\n"
                )
                info_file.write("\n")

    print(f".info file saved at: {info_file_path}")


def save_short_info_file(
    marker_labels,
    marker_freq,
    analog_labels,
    analog_units,
    analog_freq,
    dir_name,
    file_name,
):
    """
    Save a simplified version of the info file with only the main parameters and headers.
    """
    print(f"Saving short info file for {file_name}")
    short_info_file_path = Path(dir_name) / f"{file_name}_short.info"
    # Use encoding='utf-8' and ignore errors
    with open(short_info_file_path, "w", encoding="utf-8", errors="ignore") as f:
        f.write(f"Marker frequency: {marker_freq} Hz\n")
        f.write(f"Analog frequency: {analog_freq} Hz\n\n")
        f.write("Marker labels:\n")
        for label in marker_labels:
            f.write(f"{label}\n")
        f.write("\nAnalog labels and units:\n")
        for label, unit in zip(analog_labels, analog_units):
            f.write(f"{label} ({unit})\n")

    print(f"Short info file saved at: {short_info_file_path}")


def save_events(datac3d, file_name, output_dir):
    """
    Save events data from the C3D file into a CSV file, including the frame number.
    """
    print(f"Saving events for {file_name}")

    # Verify if the necessary parameters for events are available
    if "EVENT" not in datac3d["parameters"]:
        print(f"No events found for {file_name}, saving empty file.")
        save_empty_file(Path(output_dir) / f"{file_name}_events.csv")
        return

    event_params = datac3d["parameters"]["EVENT"]
    required_keys = ["CONTEXTS", "LABELS", "TIMES"]
    if not all(key in event_params for key in required_keys):
        print(f"Event parameters incomplete for {file_name}, saving empty file.")
        save_empty_file(Path(output_dir) / f"{file_name}_events.csv")
        return

    # Collect the event data
    event_contexts = event_params["CONTEXTS"]["value"]
    event_labels = event_params["LABELS"]["value"]
    event_times = event_params["TIMES"]["value"][1, :]  # Only the times (line 1)
    marker_freq = datac3d["header"]["points"]["frame_rate"]

    # Build the event data
    events_data = []
    for context, label, time in zip(event_contexts, event_labels, event_times):
        frame = int(round(time * marker_freq))
        events_data.append({"Context": context, "Label": label, "Time": time, "Frame": frame})

    # Save to a CSV file
    events_df = pd.DataFrame(events_data)
    events_file_path = Path(output_dir) / f"{file_name}_events.csv"
    events_df.to_csv(events_file_path, index=False)
    print(f"Events CSV saved at: {events_file_path}")


def importc3d(dat):
    """
    Import C3D file data and parameters.
    """
    # Print the directory and name of the script being executed
    print(f"Running script: {Path(__file__).name}")
    print(f"Script directory: {Path(__file__).parent}")
    print("Running C3D to CSV conversion")
    print("================================================")
    # Load the C3D file with force platform data extraction
    datac3d = c3d(dat, extract_forceplat_data=True)
    print(f"\nProcessing file: {dat}")
    print("================================================")
    print(f"Number of markers = {datac3d['parameters']['POINT']['USED']['value'][0]}")

    point_data = datac3d["data"]["points"]
    points_residuals = datac3d["data"]["meta_points"]["residuals"]
    analogs = datac3d["data"]["analogs"]
    marker_labels = datac3d["parameters"]["POINT"]["LABELS"]["value"]
    analog_labels = datac3d["parameters"]["ANALOG"]["LABELS"]["value"]
    analog_units = (
        datac3d["parameters"]["ANALOG"]
        .get("UNITS", {})
        .get("value", ["Unknown"] * len(analog_labels))
    )

    # Check if there are point data
    if datac3d["parameters"]["POINT"]["USED"]["value"][0] > 0:
        markers = point_data[0:3, :, :].T.reshape(-1, len(marker_labels) * 3)
    else:
        markers = np.array([])  # Use an empty NumPy array if no markers

    marker_freq = datac3d["header"]["points"]["frame_rate"]
    analog_freq = datac3d["header"]["analogs"]["frame_rate"]

    # Check for force platform data
    has_platforms = "platform" in datac3d["data"]
    num_platforms = len(datac3d["data"]["platform"]) if has_platforms else 0

    # Print summary information
    num_analog_channels = datac3d["parameters"]["ANALOG"]["USED"]["value"][0]
    print(f"Number of marker labels = {len(marker_labels)}")
    print(f"Number of analog channels = {num_analog_channels}")
    print(f"Number of force platforms = {num_platforms}")
    print(f"Marker frequency = {marker_freq} Hz")
    print(f"Analog frequency = {analog_freq} Hz")

    return (
        markers,
        marker_labels,
        marker_freq,
        analogs,
        points_residuals,
        analog_labels,
        analog_units,
        analog_freq,
        datac3d,
    )


def save_empty_file(file_path):
    """
    Save an empty CSV file.
    """
    print(f"Saving empty file: {file_path}")
    with open(file_path, "w") as f:
        f.write("")


def save_platform_data(datac3d, file_name, output_dir):
    """
    Save force platform data including Center of Pressure (COP) from the C3D file into CSV files.
    """
    print(f"Checking for platform data in {file_name}")

    if "platform" not in datac3d["data"] or not datac3d["data"]["platform"]:
        print(f"No force platform data found for {file_name}")
        save_empty_file(Path(output_dir) / f"{file_name}_cop_all.csv")
        return

    platforms = datac3d["data"]["platform"]
    num_platforms = len(platforms)
    print(f"Found {num_platforms} force platforms for {file_name}")

    # Get analog frequency for time column
    analog_freq = datac3d["header"]["analogs"]["frame_rate"]

    # Prepare combined dataframe for all COPs
    all_cop_data = []
    max_frames = 0

    # Process each platform
    for platform_idx, platform in enumerate(platforms):
        print(f"Processing platform {platform_idx} for {file_name}")

        # Extract and save center of pressure data
        if "center_of_pressure" in platform:
            cop_data = platform["center_of_pressure"]
            num_frames = cop_data.shape[1]
            max_frames = max(max_frames, num_frames)

            # Create DataFrame with time and cop x, y, z
            time_precision = get_time_precision(analog_freq)
            time_values = [f"{i / analog_freq:.{time_precision}f}" for i in range(num_frames)]

            cop_df = pd.DataFrame(
                {
                    "Time": time_values,
                    f"COP_X_P{platform_idx}": cop_data[0, :],
                    f"COP_Y_P{platform_idx}": cop_data[1, :],
                    f"COP_Z_P{platform_idx}": cop_data[2, :],
                }
            )

            # Save individual platform COP data to CSV
            cop_file_path = Path(output_dir) / f"{file_name}_platform{platform_idx}_cop.csv"
            cop_df.to_csv(cop_file_path, index=False)
            print(f"COP data saved to: {cop_file_path}")

            # Add to combined data
            all_cop_data.append({"platform": platform_idx, "data": cop_data})

        # Extract and save force data
        if "force" in platform:
            force_data = platform["force"]

            # Create DataFrame with time and force x, y, z
            num_frames = force_data.shape[1]
            time_precision = get_time_precision(analog_freq)
            time_values = [f"{i / analog_freq:.{time_precision}f}" for i in range(num_frames)]

            force_df = pd.DataFrame(
                {
                    "Time": time_values,
                    "Force_X": force_data[0, :],
                    "Force_Y": force_data[1, :],
                    "Force_Z": force_data[2, :],
                }
            )

            # Save to CSV
            force_file_path = Path(output_dir) / f"{file_name}_platform{platform_idx}_force.csv"
            force_df.to_csv(force_file_path, index=False)
            print(f"Force data saved to: {force_file_path}")

        # Extract and save moment data
        if "moment" in platform:
            moment_data = platform["moment"]

            # Create DataFrame with time and moment x, y, z
            num_frames = moment_data.shape[1]
            time_precision = get_time_precision(analog_freq)
            time_values = [f"{i / analog_freq:.{time_precision}f}" for i in range(num_frames)]

            moment_df = pd.DataFrame(
                {
                    "Time": time_values,
                    "Moment_X": moment_data[0, :],
                    "Moment_Y": moment_data[1, :],
                    "Moment_Z": moment_data[2, :],
                }
            )

            # Save to CSV
            moment_file_path = Path(output_dir) / f"{file_name}_platform{platform_idx}_moment.csv"
            moment_df.to_csv(moment_file_path, index=False)
            print(f"Moment data saved to: {moment_file_path}")

    # Create combined COP CSV with data from all platforms
    if all_cop_data:
        # Create time column
        time_precision = get_time_precision(analog_freq)
        time_values = [f"{i / analog_freq:.{time_precision}f}" for i in range(max_frames)]
        combined_cop = {"Time": time_values}

        # Add data from each platform
        for platform_data in all_cop_data:
            platform_idx = platform_data["platform"]
            cop_data = platform_data["data"]
            num_frames = cop_data.shape[1]

            # Add columns for this platform
            combined_cop[f"P{platform_idx}_COP_X"] = np.pad(
                cop_data[0, :],
                (0, max_frames - num_frames),
                "constant",
                constant_values=np.nan,
            )
            combined_cop[f"P{platform_idx}_COP_Y"] = np.pad(
                cop_data[1, :],
                (0, max_frames - num_frames),
                "constant",
                constant_values=np.nan,
            )
            combined_cop[f"P{platform_idx}_COP_Z"] = np.pad(
                cop_data[2, :],
                (0, max_frames - num_frames),
                "constant",
                constant_values=np.nan,
            )

        # Save combined COP data
        combined_cop_df = pd.DataFrame(combined_cop)
        combined_cop_path = Path(output_dir) / f"{file_name}_cop_all.csv"
        combined_cop_df.to_csv(combined_cop_path, index=False)
        print(f"Combined COP data saved to: {combined_cop_path}")

    # Save a summary file with platform information
    try:
        platform_info_path = Path(output_dir) / f"{file_name}_platform_info.csv"
        platform_info = []

        for platform_idx, platform in enumerate(platforms):
            if "origin" in platform and "corners" in platform:
                origin = platform["origin"]
                corners = platform["corners"]

                platform_info.append(
                    {
                        "Platform": platform_idx,
                        "Origin_X": origin[0],
                        "Origin_Y": origin[1],
                        "Origin_Z": origin[2],
                        "Corner1_X": corners[0, 0],
                        "Corner1_Y": corners[1, 0],
                        "Corner2_X": corners[0, 1],
                        "Corner2_Y": corners[1, 1],
                        "Corner3_X": corners[0, 2],
                        "Corner3_Y": corners[1, 2],
                        "Corner4_X": corners[0, 3],
                        "Corner4_Y": corners[1, 3],
                    }
                )

        if platform_info:
            pd.DataFrame(platform_info).to_csv(platform_info_path, index=False)
            print(f"Platform information saved to: {platform_info_path}")
    except Exception:
        print("Error saving platform information")


def save_rotation_data(datac3d, file_name, output_dir):
    """Save rotation data if available in the C3D file"""
    print(f"Checking for rotation data in {file_name}")

    rotations = datac3d["header"]["rotations"]
    if rotations["size"] > 0 and "rotations" in datac3d["data"]:
        rotation_data = datac3d["data"]["rotations"]
        # Process and save rotation data...
        rotation_df = pd.DataFrame(rotation_data)
        rotation_path = Path(output_dir) / f"{file_name}_rotations.csv"
        rotation_df.to_csv(rotation_path, index=False)
        print(f"Rotation data saved to: {rotation_path}")
    else:
        print(f"No rotation data found for {file_name}")
        save_empty_file(Path(output_dir) / f"{file_name}_rotations.csv")


def save_meta_points_data(datac3d, file_name, output_dir):
    """Save all meta_points data from the C3D file"""
    print(f"Checking for meta_points data in {file_name}")

    if "meta_points" in datac3d["data"]:
        meta_points = datac3d["data"]["meta_points"]
        marker_freq = datac3d["header"]["points"]["frame_rate"]

        # Already saving residuals, check for other metadata
        for meta_key, meta_data in meta_points.items():
            if meta_key != "residuals":  # Skip residuals as it's already saved
                print(f"Processing meta_points {meta_key}, shape: {meta_data.shape}")

                try:
                    # Handle different dimensions for meta_data
                    if len(meta_data.shape) == 3:
                        # For 3D data, save each "layer" separately
                        for i in range(meta_data.shape[0]):
                            layer_data = meta_data[i, :, :].T  # Transpose to get frames as rows

                            # Create time column
                            time_precision = get_time_precision(marker_freq)
                            time_values = [
                                f"{j / marker_freq:.{time_precision}f}"
                                for j in range(layer_data.shape[0])
                            ]

                            # Create DataFrame with appropriate column names
                            cols = [f"{meta_key}_{i}_{j}" for j in range(layer_data.shape[1])]
                            df = pd.DataFrame(layer_data, columns=cols)
                            df.insert(0, "Time", time_values)

                            # Save to CSV
                            meta_path = (
                                Path(output_dir) / f"{file_name}_meta_{meta_key}_layer{i}.csv"
                            )
                            df.to_csv(meta_path, index=False)
                            print(f"Meta points {meta_key} layer {i} saved to: {meta_path}")

                    elif len(meta_data.shape) == 2:
                        # For 2D data, can convert directly
                        time_precision = get_time_precision(marker_freq)
                        time_values = [
                            f"{j / marker_freq:.{time_precision}f}"
                            for j in range(meta_data.shape[1])
                        ]
                        df = pd.DataFrame(meta_data.T)  # Transpose to get frames as rows
                        df.insert(0, "Time", time_values)
                        meta_path = Path(output_dir) / f"{file_name}_meta_{meta_key}.csv"
                        df.to_csv(meta_path, index=False)
                        print(f"Meta points {meta_key} saved to: {meta_path}")

                    else:
                        print(
                            f"Skipping meta_points {meta_key}: unsupported shape {meta_data.shape}"
                        )

                except Exception as e:
                    print(f"Error processing meta_points {meta_key}: {e}")


def save_header_summary(datac3d, file_name, output_dir):
    """Save a summary of all header information"""
    print(f"Saving header summary for {file_name}")

    header_path = Path(output_dir) / f"{file_name}_header.csv"

    # Flatten header structure into rows
    header_rows = []
    for section, content in datac3d["header"].items():
        if isinstance(content, dict):
            for key, value in content.items():
                header_rows.append({"Section": section, "Property": key, "Value": str(value)})
        else:
            header_rows.append({"Section": section, "Property": "", "Value": str(content)})

    pd.DataFrame(header_rows).to_csv(header_path, index=False)
    print(f"Header summary saved to: {header_path}")


def save_parameter_groups(datac3d, file_name, output_dir):
    """Save important parameter groups to separate files for easy access"""
    print(f"Saving parameter groups for {file_name}")

    # Important parameter groups that users often need
    key_groups = ["POINT", "ANALOG", "FORCE_PLATFORM", "TRIAL", "SUBJECT"]

    for group in key_groups:
        if group in datac3d["parameters"]:
            group_path = Path(output_dir) / f"{file_name}_params_{group}.csv"

            # Flatten parameter structure into rows
            param_rows = []
            for param, content in datac3d["parameters"][group].items():
                if "value" in content:
                    value = content["value"]
                    # Convert arrays and lists to strings
                    if isinstance(value, (np.ndarray, list, tuple)):
                        try:
                            # For numeric arrays, show shape and sample
                            if isinstance(value, np.ndarray) and value.size > 10:
                                value_str = (
                                    f"Array shape {value.shape}, sample: {value.flatten()[:5]}..."
                                )
                            else:
                                value_str = str(value)
                        except Exception:
                            value_str = "Array (could not convert to string)"
                    else:
                        value_str = str(value)

                    param_rows.append(
                        {
                            "Parameter": param,
                            "Value": value_str,
                            "Description": content.get("description", ""),
                        }
                    )

            if param_rows:
                pd.DataFrame(param_rows).to_csv(group_path, index=False)
                print(f"Parameter group {group} saved to: {group_path}")


def save_data_statistics(datac3d, file_name, output_dir):
    """Save statistics about the data in the C3D file"""
    print(f"Calculating data statistics for {file_name}")

    stats = {
        "File": file_name,
        "Markers_Count": datac3d["parameters"]["POINT"]["USED"]["value"][0],
        "Analog_Channels": datac3d["parameters"]["ANALOG"]["USED"]["value"][0],
        "Frame_Count": datac3d["header"]["points"]["last_frame"]
        - datac3d["header"]["points"]["first_frame"]
        + 1,
        "First_Frame": datac3d["header"]["points"]["first_frame"],
        "Last_Frame": datac3d["header"]["points"]["last_frame"],
        "Duration_Seconds": (
            datac3d["header"]["points"]["last_frame"]
            - datac3d["header"]["points"]["first_frame"]
            + 1
        )
        / datac3d["header"]["points"]["frame_rate"],
        "Marker_Rate": datac3d["header"]["points"]["frame_rate"],
        "Analog_Rate": datac3d["header"]["analogs"]["frame_rate"],
        "Platforms_Count": (
            len(datac3d["data"]["platform"]) if "platform" in datac3d["data"] else 0
        ),
    }

    # Try to get more details if available
    try:
        if (
            "TRIAL" in datac3d["parameters"]
            and "ACTUAL_START_FIELD" in datac3d["parameters"]["TRIAL"]
        ):
            stats["Trial_Start"] = str(
                datac3d["parameters"]["TRIAL"]["ACTUAL_START_FIELD"]["value"]
            )
        if (
            "TRIAL" in datac3d["parameters"]
            and "ACTUAL_END_FIELD" in datac3d["parameters"]["TRIAL"]
        ):
            stats["Trial_End"] = str(datac3d["parameters"]["TRIAL"]["ACTUAL_END_FIELD"]["value"])
        if "SUBJECT" in datac3d["parameters"] and "NAME" in datac3d["parameters"]["SUBJECT"]:
            stats["Subject"] = str(datac3d["parameters"]["SUBJECT"]["NAME"]["value"])
    except Exception:
        pass

    stats_path = Path(output_dir) / f"{file_name}_statistics.csv"
    pd.DataFrame([stats]).to_csv(stats_path, index=False)
    print(f"Data statistics saved to: {stats_path}")


def save_to_files(
    markers,
    marker_labels,
    marker_freq,
    analogs,
    points_residuals,
    analog_labels,
    analog_units,
    analog_freq,
    file_name,
    run_save_dir,
    save_excel,
    datac3d,
):
    """
    Save the extracted data to CSV files and .info files within a specific directory
    created for the specific file, within the execution save directory.
    """
    print(f"Saving data to files for {file_name}")
    # Create a subfolder for the file within the execution save directory
    file_dir = Path(run_save_dir) / file_name
    file_dir.mkdir(parents=True, exist_ok=True)
    print(f"Directory created: {file_dir}")

    # Save detailed .info file and the short .info file
    save_info_file(datac3d, file_name, str(file_dir))
    save_short_info_file(
        marker_labels,
        marker_freq,
        analog_labels,
        analog_units,
        analog_freq,
        str(file_dir),
        file_name,
    )
    # Save events
    save_events(datac3d, file_name, str(file_dir))

    # Save force platform data
    save_platform_data(datac3d, file_name, str(file_dir))

    # Save rotation data
    save_rotation_data(datac3d, file_name, str(file_dir))

    # Save meta_points data
    save_meta_points_data(datac3d, file_name, str(file_dir))

    # Save header summary
    save_header_summary(datac3d, file_name, str(file_dir))

    # Save parameter groups
    save_parameter_groups(datac3d, file_name, str(file_dir))

    # Save data statistics
    save_data_statistics(datac3d, file_name, str(file_dir))

    # Prepare marker columns
    marker_columns = [f"{label}_{axis}" for label in marker_labels for axis in ["X", "Y", "Z"]]

    # Save markers data
    if markers.size > 0:
        markers_df = pd.DataFrame(markers, columns=marker_columns)
        time_precision = get_time_precision(marker_freq)
        markers_df.insert(
            0,
            "Time",
            pd.Series(
                [f"{i / marker_freq:.{time_precision}f}" for i in range(markers_df.shape[0])],
                name="Time",
            ),
        )
        print(f"Saving markers CSV for {file_name}")
        markers_df.to_csv(file_dir / f"{file_name}_markers.csv", index=False)
    else:
        print(f"No markers found for {file_name}, saving empty file.")
        save_empty_file(file_dir / f"{file_name}_markers.csv")

    # Save analog data
    if analogs.size > 0:
        analogs_df = pd.DataFrame(analogs.squeeze(axis=0).T, columns=analog_labels)
        time_precision = get_time_precision(analog_freq)
        analogs_df.insert(
            0,
            "Time",
            pd.Series(
                [f"{i / analog_freq:.{time_precision}f}" for i in range(analogs_df.shape[0])],
                name="Time",
            ),
        )
        print(f"Saving analogs CSV for {file_name}")
        analogs_df.to_csv(file_dir / f"{file_name}_analogs.csv", index=False)
    else:
        print(f"No analogs found for {file_name}, saving empty file.")
        save_empty_file(file_dir / f"{file_name}_analogs.csv")

    # Save points residuals data
    if points_residuals.size > 0:
        points_residuals_df = pd.DataFrame(points_residuals.squeeze(axis=0).T)
        time_precision = get_time_precision(marker_freq)
        points_residuals_df.insert(
            0,
            "Time",
            pd.Series(
                [
                    f"{i / marker_freq:.{time_precision}f}"
                    for i in range(points_residuals_df.shape[0])
                ],
                name="Time",
            ),
        )
        print(f"Saving points residuals CSV for {file_name}")
        points_residuals_df.to_csv(file_dir / f"{file_name}_points_residuals.csv", index=False)
    else:
        print(f"No points residuals found for {file_name}, saving empty file.")
        save_empty_file(file_dir / f"{file_name}_points_residuals.csv")

    # Optionally save to Excel
    if save_excel:
        print("Saving to Excel. This process can take a long time...")
        with pd.ExcelWriter(file_dir / f"{file_name}.xlsx", engine="openpyxl") as writer:
            if markers.size > 0:
                markers_df.to_excel(writer, sheet_name="Markers", index=False)
            if analogs.size > 0:
                analogs_df.to_excel(writer, sheet_name="Analogs", index=False)
            if points_residuals.size > 0:
                points_residuals_df.to_excel(writer, sheet_name="Points Residuals", index=False)

            # Add platform data to Excel if available
            if "platform" in datac3d["data"] and datac3d["data"]["platform"]:
                for platform_idx, platform in enumerate(datac3d["data"]["platform"]):
                    if "center_of_pressure" in platform:
                        cop_data = platform["center_of_pressure"]
                        num_frames = cop_data.shape[1]
                        time_precision = get_time_precision(analog_freq)
                        time_values = [
                            f"{i / analog_freq:.{time_precision}f}" for i in range(num_frames)
                        ]

                        cop_df = pd.DataFrame(
                            {
                                "Time": time_values,
                                "COP_X": cop_data[0, :],
                                "COP_Y": cop_data[1, :],
                                "COP_Z": cop_data[2, :],
                            }
                        )
                        cop_df.to_excel(
                            writer,
                            sheet_name=f"Platform{platform_idx}_COP",
                            index=False,
                        )

    print(f"Files for {file_name} saved successfully!")
    return str(file_dir)


def convert_c3d_to_csv():
    """
    Main function to convert C3D files to CSV and .info files.
    """
    print("=" * 60)
    print("SINGLE C3D TO CSV CONVERSION")
    print("=" * 60)
    print(f"Running script: {Path(__file__).name}")
    print(f"Script directory: {Path(__file__).parent}")
    print("Starting SINGLE file processing mode...")
    print("=" * 60)

    root = Tk()
    root.withdraw()

    print("Step 1: Getting user parameters...")
    save_excel = messagebox.askyesno(
        "Save as Excel",
        "Do you want to save the data as Excel files? This process can be very slow.",
    )
    print(f"Excel export: {'Yes' if save_excel else 'No'}")

    print("Step 2: Selecting input directory...")
    input_directory = filedialog.askdirectory(title="Select Input Directory with C3D Files")
    if not input_directory:
        print("No input directory selected. Exiting.")
        messagebox.showerror("Error", "No input directory selected.")
        root.destroy()
        return
    print(f"Input directory selected: {input_directory}")

    print("Step 3: Selecting output directory...")
    output_directory = filedialog.askdirectory(title="Select Output Directory")
    if not output_directory:
        print("No output directory selected. Exiting.")
        messagebox.showerror("Error", "No output directory selected.")
        root.destroy()
        return
    print(f"Output directory selected: {output_directory}")

    print("Step 4: Scanning for C3D files...")
    # Get all C3D files in the input directory (excluding hidden files that start with '.')
    input_path = Path(input_directory)
    c3d_files = [
        f.name
        for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() == ".c3d" and not f.name.startswith(".")
    ]

    if not c3d_files:
        print(f"ERROR: No visible C3D files found in {input_directory}")
        messagebox.showerror("Error", f"No visible C3D files found in {input_directory}")
        root.destroy()
        return

    print(f"Found {len(c3d_files)} visible C3D files in directory")
    for f in c3d_files:
        print(f"  - {f}")

    print("Step 5: Creating output directory...")
    # Create the root directory for saving with timestamp in the name
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_save_dir = Path(output_directory) / f"vaila_c3d_to_csv_{run_timestamp}"
    run_save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run-level save directory created: {run_save_dir}")

    print("Step 6: Starting processing...")
    print("=" * 50)

    # Initialize tracking variables
    successful_conversions = 0
    failed_conversions = 0
    error_details = []

    progress_bar = tqdm(total=len(c3d_files), desc="Processing C3D files", unit="file")

    for c3d_file in c3d_files:
        print(f"\nProcessing file: {c3d_file}")
        try:
            file_path = Path(input_directory) / c3d_file
            (
                markers,
                marker_labels,
                marker_freq,
                analogs,
                points_residuals,
                analog_labels,
                analog_units,
                analog_freq,
                datac3d,
            ) = importc3d(str(file_path))
            file_name = Path(c3d_file).stem

            # Save the extracted data in CSV files and .info files within the execution save directory
            save_to_files(
                markers,
                marker_labels,
                marker_freq,
                analogs,
                points_residuals,
                analog_labels,
                analog_units,
                analog_freq,
                file_name,
                str(run_save_dir),
                save_excel,
                datac3d,
            )
            successful_conversions += 1
            print(f"Successfully converted: {c3d_file}")

        except Exception as e:
            failed_conversions += 1
            error_details.append((c3d_file, str(e)))
            print(f"ERROR processing {c3d_file}: {e}")
            messagebox.showerror("Error", f"Failed to process {c3d_file}: {e}")

        progress_bar.update(1)

    progress_bar.close()

    # Show final results
    print(f"\n{'=' * 60}")
    print("SINGLE CONVERSION COMPLETED")
    print(f"{'=' * 60}")
    print(f"Visible C3D files found: {len(c3d_files)}")
    print(f"Successful conversions: {successful_conversions}")
    print(f"Failed conversions: {failed_conversions}")
    print(f"Success rate: {(successful_conversions / len(c3d_files) * 100):.1f}%")
    print(f"Output directory: {run_save_dir}")
    print(f"{'=' * 60}")

    if failed_conversions == 0:
        print("PERFECT! All files converted successfully!")
        messagebox.showinfo("Information", "C3D files conversion completed successfully!")
    else:
        print(f"Warning: {failed_conversions} files failed to convert.")
        messagebox.showwarning(
            "Warning",
            f"Conversion completed with {failed_conversions} failures. Check console for details.",
        )

    root.destroy()  # Close the Tkinter resources


def print_complete_data_structure(data, prefix=""):
    """Recursively explore and print all data structure keys and shapes"""
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list, np.ndarray)):
                print(f"{prefix}{key}: {type(value)}")
                print_complete_data_structure(value, prefix + "  ")
            else:
                print(f"{prefix}{key}: {value}")
    elif isinstance(data, list):
        if len(data) > 0:
            print(f"{prefix}[0]: {type(data[0])}")
            if isinstance(data[0], dict):
                print_complete_data_structure(data[0], prefix + "  ")
    elif isinstance(data, np.ndarray):
        print(f"{prefix}Shape: {data.shape}, Type: {data.dtype}")


def batch_convert_c3d_to_csv():
    """
    Main function to convert all C3D files in a directory to CSV and .info files.
    """
    print("=" * 60)
    print("BATCH C3D TO CSV CONVERSION")
    print("=" * 60)
    print(f"Running script: {Path(__file__).name}")
    print(f"Script directory: {Path(__file__).parent}")
    print("Starting BATCH processing mode...")
    print("=" * 60)

    root = Tk()
    root.withdraw()

    print("Step 1: Selecting input directory...")
    input_directory = filedialog.askdirectory(title="Select Input Directory with C3D Files")
    if not input_directory:
        print("No input directory selected. Exiting.")
        messagebox.showerror("Error", "No input directory selected.")
        return

    print(f"Input directory selected: {input_directory}")

    print("Step 2: Selecting output directory...")
    output_directory = filedialog.askdirectory(title="Select Output Directory")
    if not output_directory:
        print("No output directory selected. Exiting.")
        messagebox.showerror("Error", "No output directory selected.")
        return

    print(f"Output directory selected: {output_directory}")

    print("Step 3: Scanning for C3D files...")
    # Get all C3D files in the input directory (excluding hidden files that start with '.')
    input_path = Path(input_directory)
    c3d_files = [
        f.name
        for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() == ".c3d" and not f.name.startswith(".")
    ]

    if not c3d_files:
        print(f"ERROR: No visible C3D files found in {input_directory}")
        messagebox.showerror("Error", f"No visible C3D files found in {input_directory}")
        return

    print(f"Found {len(c3d_files)} visible C3D files in directory")
    for f in c3d_files:
        print(f"  - {f}")

    print("Step 4: Getting user parameters...")
    save_excel = messagebox.askyesno(
        "Save as Excel",
        "Do you want to save the data as Excel files? This process can be very slow.",
    )
    print(f"Excel export: {'Yes' if save_excel else 'No'}")

    print("Step 5: Creating output directory...")
    # Create the root directory for saving with timestamp in the name
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_save_dir = Path(output_directory) / f"c3d2csv_{run_timestamp}"
    run_save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run-level save directory created: {run_save_dir}")

    print("Step 6: Starting batch processing...")
    print("=" * 50)

    # Initialize tracking variables
    successful_conversions = 0
    failed_conversions = 0
    error_details = []
    successful_files = []
    failed_files = []

    # Create log file
    log_file_path = run_save_dir / "conversion_log.txt"
    log_file = open(log_file_path, "w", encoding="utf-8")

    # Write header to log file
    log_file.write("=" * 80 + "\n")
    log_file.write("C3D TO CSV BATCH CONVERSION LOG\n")
    log_file.write("=" * 80 + "\n")
    log_file.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"Input Directory: {input_directory}\n")
    log_file.write(f"Output Directory: {run_save_dir}\n")
    log_file.write(f"Excel Export: {'Yes' if save_excel else 'No'}\n")
    log_file.write(f"Visible C3D Files Found: {len(c3d_files)}\n")
    log_file.write("=" * 80 + "\n\n")

    # Process each C3D file
    for i, c3d_file in enumerate(c3d_files, 1):
        print(f"\nProcessing file {i}/{len(c3d_files)}: {c3d_file}")
        log_file.write(f"\n--- Processing File {i}/{len(c3d_files)}: {c3d_file} ---\n")

        try:
            print(f"\nProcessing: {c3d_file}")
            log_file.write("Status: Processing started\n")

            # Process the C3D file
            file_path = Path(input_directory) / c3d_file
            (
                markers,
                marker_labels,
                marker_freq,
                analogs,
                points_residuals,
                analog_labels,
                analog_units,
                analog_freq,
                datac3d,
            ) = importc3d(str(file_path))

            log_file.write("C3D file loaded successfully\n")
            log_file.write(
                f"Markers: {len(marker_labels)}, Analog channels: {len(analog_labels)}\n"
            )
            log_file.write(
                f"Marker frequency: {marker_freq} Hz, Analog frequency: {analog_freq} Hz\n"
            )

            file_name = Path(c3d_file).stem

            # Save the extracted data in CSV files and .info files within the execution save directory
            save_to_files(
                markers,
                marker_labels,
                marker_freq,
                analogs,
                points_residuals,
                analog_labels,
                analog_units,
                analog_freq,
                file_name,
                str(run_save_dir),
                save_excel,
                datac3d,
            )

            successful_conversions += 1
            successful_files.append(c3d_file)
            print(f"Successfully converted: {c3d_file}")
            log_file.write("Status: SUCCESS - All files created successfully\n")

        except Exception as e:
            failed_conversions += 1
            failed_files.append(c3d_file)
            error_msg = str(e)
            error_details.append((c3d_file, error_msg))

            print(f"ERROR processing {c3d_file}: {e}")
            print("Continuing with next file...")

            log_file.write(f"Status: FAILED - Error: {error_msg}\n")
            log_file.write(f"Error type: {type(e).__name__}\n")

            # Add more context for common errors
            if "utf-8" in error_msg.lower():
                log_file.write("Context: This appears to be a UTF-8 encoding issue\n")
            elif "keyerror" in error_msg.lower():
                log_file.write("Context: This appears to be a parameter/key access issue\n")
            elif "shape" in error_msg.lower():
                log_file.write("Context: This appears to be a data shape/dimension issue\n")
            elif "ezc3d" in error_msg.lower():
                log_file.write("Context: This appears to be a C3D file reading issue\n")

            continue

    # Write summary to log file
    log_file.write("\n" + "=" * 80 + "\n")
    log_file.write("CONVERSION SUMMARY\n")
    log_file.write("=" * 80 + "\n")
    log_file.write(f"Visible C3D files found: {len(c3d_files)}\n")
    log_file.write(f"Successful conversions: {successful_conversions}\n")
    log_file.write(f"Failed conversions: {failed_conversions}\n")
    log_file.write(f"Success rate: {(successful_conversions / len(c3d_files) * 100):.1f}%\n")

    if successful_files:
        log_file.write(f"\nSUCCESSFUL CONVERSIONS ({len(successful_files)}):\n")
        for file in successful_files:
            log_file.write(f"  ✓ {file}\n")

    if failed_files:
        log_file.write(f"\nFAILED CONVERSIONS ({len(failed_files)}):\n")
        for file, error in error_details:
            log_file.write(f"  ✗ {file} - Error: {error}\n")

    # Analyze error patterns
    if error_details:
        log_file.write("\nERROR ANALYSIS:\n")
        error_types = {}
        for file, error in error_details:
            error_type = type(error).__name__ if hasattr(error, "__class__") else "Unknown"
            if error_type not in error_types:
                error_types[error_type] = []
            error_types[error_type].append(file)

        for error_type, files in error_types.items():
            log_file.write(f"  {error_type}: {len(files)} files\n")
            for file in files:
                log_file.write(f"    - {file}\n")

    log_file.write(f"\nOutput directory: {run_save_dir}\n")
    log_file.write(f"Log file: {log_file_path}\n")
    log_file.write("=" * 80 + "\n")
    log_file.close()

    print(f"\nDetailed log saved to: {log_file_path}")

    # Show final results
    print(f"\n{'=' * 60}")
    print("BATCH CONVERSION COMPLETED")
    print(f"{'=' * 60}")
    print(f"Visible C3D files found: {len(c3d_files)}")
    print(f"Successful conversions: {successful_conversions}")
    print(f"Failed conversions: {failed_conversions}")
    print(f"Success rate: {(successful_conversions / len(c3d_files) * 100):.1f}%")
    print(f"Output directory: {run_save_dir}")
    print(f"Detailed log: {log_file_path}")
    print(f"{'=' * 60}")

    if failed_conversions == 0:
        print("PERFECT! All files converted successfully!")
    elif successful_conversions > failed_conversions:
        print("Good! Most files converted successfully.")
    else:
        print("Warning: Many files failed to convert.")

    # Show error summary if there were failures
    if failed_conversions > 0:
        print("\nERROR SUMMARY:")
        print(f"Failed files: {failed_conversions}")
        print("Most common errors:")

        error_counts = {}
        for file, error in error_details:
            error_msg = str(error)
            if error_msg not in error_counts:
                error_counts[error_msg] = 0
            error_counts[error_msg] += 1

        # Show top 5 most common errors
        sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
        for i, (error_msg, count) in enumerate(sorted_errors[:5]):
            print(
                f"  {i + 1}. {error_msg[:100]}{'...' if len(error_msg) > 100 else ''} ({count} files)"
            )

    message = f"Batch conversion completed!\n\nVisible C3D files: {len(c3d_files)}\nSuccessful: {successful_conversions}\nFailed: {failed_conversions}\nSuccess rate: {(successful_conversions / len(c3d_files) * 100):.1f}%\n\nOutput directory: {run_save_dir}\nDetailed log: {log_file_path.name}"
    messagebox.showinfo("Batch Conversion Complete", message)

    root.destroy()  # Close the Tkinter resources


if __name__ == "__main__":
    print("Starting C3D Export & Inspection Tool...")
    print(f"Running: {pathlib.Path(__file__).name}")
    print(f"Directory: {pathlib.Path(__file__).parent.resolve()}")

    # Create Main Menu GUI
    def open_menu():
        root = Tk()
        root.title("C3D Export & Inspection Tool")
        root.geometry("500x400")

        # Center window
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x_c = int((screen_width / 2) - (500 / 2))
        y_c = int((screen_height / 2) - (400 / 2))
        root.geometry(f"500x400+{x_c}+{y_c}")

        # Title
        label = tk.Label(root, text="Select Mode", font=("Arial", 16, "bold"))
        label.pack(pady=20)

        # Buttons
        def run_batch():
            root.destroy()
            batch_convert_c3d_to_csv()

        def run_single():
            root.destroy()
            convert_c3d_to_csv()

        def run_inspect():
            print("Starting C3D Inspection Tool...")
            print(f"Running: {pathlib.Path(__file__).name}")
            print(f"Directory: {pathlib.Path(__file__).parent.resolve()}")
            # Launch the inspection tool
            root.destroy()
            inspect_c3d_gui()

        btn_font = ("Arial", 12)

        tk.Button(
            root, text="Batch Convert (Directory)", command=run_batch, font=btn_font, width=25
        ).pack(pady=10)
        tk.Button(
            root, text="Single Convert (File)", command=run_single, font=btn_font, width=25
        ).pack(pady=10)
        tk.Button(
            root,
            text="Inspect C3D File",
            command=run_inspect,
            font=btn_font,
            width=25,
            bg="#e1f5fe",
        ).pack(pady=10)

        tk.Button(root, text="Exit", command=root.destroy, font=btn_font, width=10, fg="red").pack(
            pady=20
        )

        root.mainloop()

    open_menu()
