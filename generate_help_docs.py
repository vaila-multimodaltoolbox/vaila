#!/usr/bin/env python3
"""
Script to automatically generate help documentation for all Python modules in vailá.

This script:
1. Analyzes all .py files in the vaila/ directory
2. Extracts information such as docstrings, functions, version, author
3. Generates help files in HTML and MD formats
4. Organizes by functional categories

Author: vailá Automatic Documentation System
Date: 2025
"""

import os
import re
import ast
import sys
from pathlib import Path
from datetime import datetime
import json

# Categorias organizacionais para os módulos
MODULE_CATEGORIES = {
    "analysis": [
        "markerless_2d_analysis",
        "markerless_3d_analysis",
        "markerless_live",
        "cluster_analysis",
        "mocap_analysis",
        "imu_analysis",
        "forceplate_analysis",
        "emg_labiocom",
        "gnss_analysis",
        "animal_open_field",
        "vaila_and_jump",
        "cube2d_kinematics",
        "vector_coding",
        "run_vector_coding",
    ],
    "processing": [
        "readc3d_export",
        "readcsv_export",
        "readcsv",
        "rearrange_data",
        "interp_smooth_split",
        "filtering",
        "filter_utils",
        "dlt2d",
        "dlt3d",
        "rec2d",
        "rec2d_one_dlt2d",
        "rec3d",
        "rec3d_one_dlt3d",
        "reid_markers",
        "modifylabref",
        "data_processing",
    ],
    "visualization": [
        "vailaplot2d",
        "vailaplot3d",
        "viewc3d",
        "showc3d",
        "soccerfield",
        "plotting",
    ],
    "ml": [
        "yolov11track",
        "yolov12track",
        "yolotrain",
        "vaila_mlwalkway",
        "ml_models_training",
        "ml_valid_models",
        "walkway_ml_prediction",
        "markerless2d_mpyolo",
        "markerless2d_analysis_v2",
        "markerless3d_analysis_v2",
    ],
    "tools": [
        "filemanager",
        "compress_videos_h264",
        "compress_videos_h265",
        "compress_videos_h266",
        "videoprocessor",
        "extractpng",
        "cutvideo",
        "resize_video",
        "getpixelvideo",
        "numberframes",
        "syncvid",
        "drawboxe",
        "vaila_ytdown",
        "vaila_iaudiovid",
        "rm_duplicateframes",
        "vaila_upscaler",
        "vaila_lensdistortvideo",
        "vaila_distortvideo_gui",
        "vaila_datdistort",
        "cop_analysis",
        "cop_calculate",
        "force_cmj",
        "force_cube_fig",
        "grf_gait",
        "stabilogram_analysis",
        "spectral_features",
        "usound_biomec1",
        "brainstorm",
        "scout_vaila",
        "skout_bundle",
        "batchcut",
        "merge_multivideos",
        "mergestack",
        "convert_videos_ts_to_mp4",
        "getcampardistortlens",
        "usvideoia",
        "sync_flash",
        "standardize_header",
        "join2dataset",
        "load_vicon_csv_split_batch",
        "linear_interpolation_split",
        "fixnoise",
        "rotation",
        "ellipse",
        "numstepsmp",
        "process_gait_features",
    ],
    "utils": [
        "common_utils",
        "utils",
        "dialogsuser",
        "dialogsuser_cluster",
        "native_file_dialog",
        "vaila_manifest",
        "backup_markerless",
        "example_batch_usage",
        "listjointsnames",
        "reid_yolotrack",
        "reidmplrswap",
        "reidvideogui",
        "dlc2vaila",
        "modifylabref_cli",
        "vpython_c3d",
        "compressvideo",
        "markerless_2d_analysis_nvidia",
        "mphands",
        "mpangles",
    ],
}


def extract_module_info(file_path):
    """Extracts information from a Python module"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Parse AST to extract functions
        try:
            tree = ast.parse(content)
            functions = [
                node.name
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
            ]
        except:
            functions = []

        # Extract docstring information
        docstring_match = re.search(r'""".*?"""', content, re.DOTALL)
        docstring = docstring_match.group(0).strip('"""') if docstring_match else ""

        # Extract basic information
        module_info = {
            "file_path": file_path,
            "module_name": Path(file_path).stem,
            "functions": functions[:20],  # Limit to avoid very large files
            "docstring": (
                docstring[:1000] + "..." if len(docstring) > 1000 else docstring
            ),
            "has_gui": "tkinter" in content.lower() or "gui" in content.lower(),
            "file_size": len(content),
            "line_count": len(content.split("\n")),
        }

        # Try to extract version and author from docstring
        version_match = re.search(r"Version[:\s]+([^\n\r]+)", docstring, re.IGNORECASE)
        if version_match:
            module_info["version"] = version_match.group(1).strip()

        author_match = re.search(r"Author[:\s]+([^\n\r]+)", docstring, re.IGNORECASE)
        if author_match:
            module_info["author"] = author_match.group(1).strip()

        return module_info

    except Exception as e:
        return {
            "file_path": file_path,
            "module_name": Path(file_path).stem,
            "error": str(e),
            "functions": [],
            "docstring": "",
            "has_gui": False,
        }


def get_module_category(module_name):
    """Determines the category of a module"""
    for category, modules in MODULE_CATEGORIES.items():
        if module_name in modules:
            return category
    return "uncategorized"


def generate_html_help(module_info):
    """Generates HTML help file for a module"""
    category = get_module_category(module_info["module_name"])

    html_content = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>vailá - {module_info['module_name']}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
            line-height: 1.6;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 800px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .module-info {{ background: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .functions {{ background: #f8f9fa; padding: 15px; border-left: 4px solid #28a745; margin: 15px 0; }}
        .docstring {{ background: #fff3cd; padding: 15px; border-left: 4px solid #ffc107; margin: 15px 0; }}
        .error {{ background: #f8d7da; padding: 15px; border-left: 4px solid #dc3545; margin: 15px 0; }}
        code {{ background: #e9ecef; padding: 2px 6px; border-radius: 3px; }}
        pre {{ background: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #dee2e6; color: #6c757d; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>vailá - {module_info['module_name']}</h1>

        <div class="module-info">
            <h3>📋 Module Information</h3>
            <p><strong>Category:</strong> {category.title()}</p>
            <p><strong>File:</strong> {module_info['file_path']}</p>
            <p><strong>Lines:</strong> {module_info.get('line_count', 'N/A')}</p>
            <p><strong>Size:</strong> {module_info.get('file_size', 0)} characters</p>
            {f"<p><strong>Version:</strong> {module_info.get('version', 'N/A')}</p>" if module_info.get('version') else ""}
            {f"<p><strong>Author:</strong> {module_info.get('author', 'N/A')}</p>" if module_info.get('author') else ""}
            <p><strong>GUI Interface:</strong> {'✅ Yes' if module_info.get('has_gui') else '❌ No'}</p>
        </div>

        {f"""
        <div class="error">
            <h3>⚠️ Analysis Error</h3>
            <p>{module_info.get('error', 'Unknown error')}</p>
        </div>
        """ if module_info.get('error') else ""}

        <div class="docstring">
            <h3>📖 Description</h3>
            <pre>{module_info.get('docstring', 'No description available')}</pre>
        </div>

        <div class="functions">
            <h3>🔧 Main Functions</h3>
            {f"<p><strong>Total functions found:</strong> {len(module_info.get('functions', []))}</p>" if module_info.get('functions') else "<p>No functions found</p>"}
            {f"<ul>{''.join(f'<li><code>{func}</code></li>' for func in module_info.get('functions', []))}</ul>" if module_info.get('functions') else ""}
        </div>

        <div class="footer">
            <p>📅 Generated automatically on: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
            <p>🔗 Part of vailá - Multimodal Toolbox</p>
            <p>🌐 <a href="https://github.com/vaila-multimodaltoolbox/vaila">GitHub Repository</a></p>
        </div>
    </div>
</body>
</html>"""

    return html_content


def generate_md_help(module_info):
    """Generates MD help file for a module"""
    category = get_module_category(module_info["module_name"])

    md_content = f"""# {module_info['module_name']}

## 📋 Module Information

- **Category:** {category.title()}
- **File:** `{module_info['file_path']}`
- **Lines:** {module_info.get('line_count', 'N/A')}
- **Size:** {module_info.get('file_size', 0)} characters
{ f"- **Version:** {module_info.get('version', 'N/A')}" if module_info.get('version') else "" }
{ f"- **Author:** {module_info.get('author', 'N/A')}" if module_info.get('author') else "" }
- **GUI Interface:** {'✅ Yes' if module_info.get('has_gui') else '❌ No'}

## 📖 Description

{module_info.get('docstring', 'No description available')}

## 🔧 Main Functions

{ f"**Total functions found:** {len(module_info.get('functions', []))}" if module_info.get('functions') else "No functions found" }

{ ''.join(f"- `{func}`\n" for func in module_info.get('functions', [])) if module_info.get('functions') else "" }

{ f"## ⚠️ Analysis Error\n\n{module_info.get('error', 'Unknown error')}" if module_info.get('error') else "" }

---

📅 **Generated automatically on:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
🔗 **Part of vailá - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
"""

    return md_content


def main():
    """Main function to generate all help documentation"""
    print("🔍 Starting analysis of Python modules in vailá...")
    print(f"📂 Base directory: {os.getcwd()}")

    # Find all Python files in the vaila directory
    vaila_dir = Path("vaila")
    python_files = list(vaila_dir.rglob("*.py"))

    print(f"📊 Found {len(python_files)} Python files")

    # Analyze each file
    modules_info = []
    for py_file in python_files:
        print(f"🔍 Analyzing: {py_file}")
        module_info = extract_module_info(py_file)
        modules_info.append(module_info)

    print(f"✅ Analysis complete! {len(modules_info)} modules processed")

    # Generate help files
    help_dir = Path("vaila/help")
    help_dir.mkdir(exist_ok=True)

    generated_files = []

    for module_info in modules_info:
        category = get_module_category(module_info["module_name"])
        category_dir = help_dir / category
        category_dir.mkdir(exist_ok=True)

        # Generate HTML
        html_content = generate_html_help(module_info)
        html_file = category_dir / f"{module_info['module_name']}.html"

        with open(html_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        # Generate MD
        md_content = generate_md_help(module_info)
        md_file = category_dir / f"{module_info['module_name']}.md"

        with open(md_file, "w", encoding="utf-8") as f:
            f.write(md_content)

        generated_files.append(
            {
                "module": module_info["module_name"],
                "category": category,
                "html": str(html_file),
                "md": str(md_file),
            }
        )

        print(f"✅ Generated help for: {module_info['module_name']} ({category})")

    # Generate general index
    index_content = generate_index(generated_files)

    with open(help_dir / "index.html", "w", encoding="utf-8") as f:
        f.write(index_content["html"])

    with open(help_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(index_content["md"])

    print("🎉 Help documentation generated successfully!")
    print(f"📂 Files created in: {help_dir}")
    print(f"📊 Total documented modules: {len(generated_files)}")
    print(
        f"📁 Categories: {', '.join(set(item['category'] for item in generated_files))}"
    )


def generate_index(generated_files):
    """Generates general index for help documentation"""

    # Organize by category
    categories = {}
    for item in generated_files:
        cat = item["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(item)

    html_content = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>vailá - Documentação de Help</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
            line-height: 1.6;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{ color: #2c3e50; text-align: center; border-bottom: 3px solid #3498db; padding-bottom: 20px; }}
        h2 {{ color: #34495e; margin-top: 40px; border-left: 5px solid #3498db; padding-left: 15px; }}
        .category {{ margin: 20px 0; }}
        .module-list {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; margin: 15px 0; }}
        .module-card {{
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            background: #f8f9fa;
            transition: box-shadow 0.3s;
        }}
        .module-card:hover {{ box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
        .module-name {{ font-weight: bold; color: #2c3e50; margin-bottom: 10px; }}
        .module-links {{ margin-top: 10px; }}
        .module-links a {{ margin-right: 15px; text-decoration: none; padding: 5px 10px; border-radius: 3px; }}
        .html-link {{ background: #007bff; color: white; }}
        .md-link {{ background: #28a745; color: white; }}
        .footer {{ margin-top: 50px; padding-top: 20px; border-top: 1px solid #dee2e6; text-align: center; color: #6c757d; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📚 vailá - Help Documentation</h1>
        <p style="text-align: center; font-size: 1.1em; color: #666; margin-bottom: 40px;">
            Automatically generated documentation for all Python modules in vailá Multimodal Toolbox
        </p>

        <div style="text-align: center; margin: 30px 0;">
            <strong>Total documented modules:</strong> {len(generated_files)} |
            <strong>Categories:</strong> {len(categories)} |
            <strong>Generated on:</strong> {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
        </div>
"""

    for category, modules in categories.items():
        html_content += f"""
        <div class="category">
            <h2>{category.title()} ({len(modules)} modules)</h2>
            <div class="module-list">
"""

        for module in sorted(modules, key=lambda x: x["module"]):
            html_content += f"""
                <div class="module-card">
                    <div class="module-name">{module['module']}</div>
                    <div class="module-links">
                        <a href="{category}/{module['module']}.html" class="html-link">📄 HTML</a>
                        <a href="{category}/{module['module']}.md" class="md-link">📝 Markdown</a>
                    </div>
                </div>
"""

        html_content += """
            </div>
        </div>
"""

    html_content += f"""
        <div class="footer">
            <p>🔗 <a href="https://github.com/vaila-multimodaltoolbox/vaila">vailá - Multimodal Toolbox</a></p>
            <p>📧 For questions or suggestions, contact the development team</p>
        </div>
    </div>
</body>
</html>"""

    md_content = f"""# 📚 vailá - Help Documentation

Automatically generated documentation for all Python modules in vailá Multimodal Toolbox.

## 📊 General Statistics

- **Total documented modules:** {len(generated_files)}
- **Categories:** {len(categories)}
- **Generated on:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}

## 📂 Module Categories

"""

    for category, modules in categories.items():
        md_content += f"### {category.title()} ({len(modules)} modules)\n\n"

        for module in sorted(modules, key=lambda x: x["module"]):
            md_content += f"- **{module['module']}**\n"
            md_content += f"  - [📄 HTML]({category}/{module['module']}.html)\n"
            md_content += f"  - [📝 Markdown]({category}/{module['module']}.md)\n\n"

    md_content += """
## 🔗 Useful Links

- [🌐 GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
- [📖 Main Documentation](https://vaila.readthedocs.io/)
- [🛠️ Issues and Discussions](https://github.com/vaila-multimodaltoolbox/vaila/issues)

## 📝 About This Documentation

This documentation was automatically generated through analysis of vailá's Python modules. Each help file contains:

- Basic module information (author, version, category)
- Description extracted from docstring
- List of main functions found
- Links to HTML and Markdown formats

To update this documentation, run the `generate_help_docs.py` script again.
"""

    return {"html": html_content, "md": md_content}

# Portuguese to English translations dictionary (shared by all translation functions)
translations = {
    # Module Information section
    "📋 Informações do Módulo": "📋 Module Information",
    "Categoria:": "Category:",
    "Arquivo:": "File:",
    "Linhas:": "Lines:",
    "Tamanho:": "Size:",
    "Versão:": "Version:",
    "Autor:": "Author:",
    "Interface Gráfica:": "GUI Interface:",
    "Sim": "Yes",
    "Não": "No",

    # Description section
    "📖 Descrição": "📖 Description",
    "Sem descrição disponível": "No description available",

    # Functions section
    "🔧 Funções Principais": "🔧 Main Functions",
    "Total de funções encontradas:": "Total functions found:",
    "Nenhuma função encontrada": "No functions found",
    "Total de funções encontradas:": "Total functions found:",

    # Error section
    "⚠️ Erro na Análise": "⚠️ Analysis Error",
    "Erro desconhecido": "Unknown error",

    # Footer
    "📅 Gerado automaticamente em:": "📅 Generated automatically on:",
    "🔗 Parte do vailá - Multimodal Toolbox": "🔗 Part of vailá - Multimodal Toolbox",
    "🌐 GitHub Repository": "🌐 GitHub Repository",
    "📧 Para dúvidas ou sugestões, entre em contato com a equipe de desenvolvimento": "📧 For questions or suggestions, contact the development team",

    # Index page
    "📚 vailá - Documentação de Help": "📚 vailá - Help Documentation",
    "Documentação automática gerada para todos os módulos Python do vailá Multimodal Toolbox": "Automatically generated documentation for all Python modules in vailá Multimodal Toolbox",
    "Total de módulos documentados:": "Total documented modules:",
    "Categorias:": "Categories:",
    "módulos": "modules",
    "módulo": "module",
    "módulos)": "modules)",
    "módulo)": "module)",
    "Gerado em:": "Generated on:",
    "Parte do vailá - Multimodal Toolbox": "Part of vailá - Multimodal Toolbox",
    "Para dúvidas ou sugestões, entre em contato com a equipe de desenvolvimento": "For questions or suggestions, contact the development team",

    # Navigation buttons
    "📄 HTML": "📄 HTML",
    "📝 Markdown": "📝 Markdown",

    # Other common terms
    "módulo": "module",
    "arquivo": "file",
    "categoria": "category",
    "linha": "line",
    "linhas": "lines",
    "tamanho": "size",
    "caracteres": "characters",
    "autor": "author",
    "versão": "version",
    "interface gráfica": "GUI interface",
    "descrição": "description",
    "função": "function",
    "funções": "functions",
    "principal": "main",
    "principais": "main",
    "erro": "error",
    "análise": "analysis",
    "total": "total",
    "encontrado": "found",
    "encontrada": "found",
    "encontradas": "found",
    "encontrado": "found",
    "nenhum": "no",
    "nenhuma": "no",
    "nenhuns": "no",
    "nenhumas": "no",
}

def translate_text(text):
    """Translate Portuguese text to English"""
    for pt, en in translations.items():
        text = text.replace(pt, en)
    return text

def translate_html_file(html_file):
    """Translate a single HTML file"""
    try:
        with open(html_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Translate content
        translated_content = translate_text(content)

        # Write back translated content
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(translated_content)

        print(f"✅ Translated: {html_file}")

    except Exception as e:
        print(f"❌ Error translating {html_file}: {e}")

def translate_md_file(md_file):
    """Translate a single MD file"""
    try:
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Translate content
        translated_content = translate_text(content)

        # Write back translated content
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(translated_content)

        print(f"✅ Translated: {md_file}")

    except Exception as e:
        print(f"❌ Error translating {md_file}: {e}")

def translate_existing_help_files():
    """Translates existing Portuguese help files to English"""
    help_dir = Path("vaila/help")

    # Find all HTML files in help directory
    html_files = list(help_dir.rglob("*.html"))
    md_files = list(help_dir.rglob("*.md"))

    print(f"🔍 Found {len(html_files)} HTML files and {len(md_files)} MD files to translate")

    # Translate HTML files
    for html_file in html_files:
        translate_html_file(str(html_file))

    # Translate MD files
    for md_file in md_files:
        translate_md_file(str(md_file))

    print("🎉 Translation complete!")

def translate_docs_files():
    """Translates existing Portuguese documentation files in docs/ to English"""
    docs_dir = Path("docs")

    # Find all MD files in docs directory (recursive)
    md_files = list(docs_dir.rglob("*.md"))

    print(f"🔍 Found {len(md_files)} MD files in docs/ to translate")

    # Translate MD files in docs
    for md_file in md_files:
        translate_md_file(str(md_file))

    print("🎉 Docs translation complete!")

if __name__ == "__main__":
    main()
    translate_existing_help_files()
    translate_docs_files()
