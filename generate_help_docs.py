#!/usr/bin/env python3
"""
Script para gerar documentação de help automaticamente para todos os módulos Python do vailá.

Este script:
1. Analisa todos os arquivos .py no diretório vaila/
2. Extrai informações como docstrings, funções, versão, autor
3. Gera arquivos de help em formato HTML e MD
4. Organiza por categorias funcionais

Autor: Sistema de Documentação Automática vailá
Data: 2025
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
    'analysis': [
        'markerless_2d_analysis', 'markerless_3d_analysis', 'markerless_live',
        'cluster_analysis', 'mocap_analysis', 'imu_analysis', 'forceplate_analysis',
        'emg_labiocom', 'gnss_analysis', 'animal_open_field', 'vaila_and_jump',
        'cube2d_kinematics', 'vector_coding', 'run_vector_coding'
    ],
    'processing': [
        'readc3d_export', 'readcsv_export', 'readcsv', 'rearrange_data',
        'interp_smooth_split', 'filtering', 'filter_utils', 'dlt2d', 'dlt3d',
        'rec2d', 'rec2d_one_dlt2d', 'rec3d', 'rec3d_one_dlt3d', 'reid_markers',
        'modifylabref', 'data_processing'
    ],
    'visualization': [
        'vailaplot2d', 'vailaplot3d', 'viewc3d', 'showc3d', 'soccerfield',
        'plotting'
    ],
    'ml': [
        'yolov11track', 'yolov12track', 'yolotrain', 'vaila_mlwalkway',
        'ml_models_training', 'ml_valid_models', 'walkway_ml_prediction',
        'markerless2d_mpyolo', 'markerless2d_analysis_v2', 'markerless3d_analysis_v2'
    ],
    'tools': [
        'filemanager', 'compress_videos_h264', 'compress_videos_h265', 'compress_videos_h266',
        'videoprocessor', 'extractpng', 'cutvideo', 'resize_video', 'getpixelvideo',
        'numberframes', 'syncvid', 'drawboxe', 'vaila_ytdown', 'vaila_iaudiovid',
        'rm_duplicateframes', 'vaila_upscaler', 'vaila_lensdistortvideo',
        'vaila_distortvideo_gui', 'vaila_datdistort', 'cop_analysis', 'cop_calculate',
        'force_cmj', 'force_cube_fig', 'grf_gait', 'stabilogram_analysis',
        'spectral_features', 'usound_biomec1', 'brainstorm', 'scout_vaila',
        'skout_bundle', 'batchcut', 'merge_multivideos', 'mergestack',
        'convert_videos_ts_to_mp4', 'getcampardistortlens', 'usvideoia',
        'sync_flash', 'standardize_header', 'join2dataset', 'load_vicon_csv_split_batch',
        'linear_interpolation_split', 'fixnoise', 'rotation', 'ellipse',
        'numstepsmp', 'process_gait_features'
    ],
    'utils': [
        'common_utils', 'utils', 'dialogsuser', 'dialogsuser_cluster',
        'native_file_dialog', 'vaila_manifest', 'backup_markerless',
        'example_batch_usage', 'listjointsnames', 'reid_yolotrack',
        'reidmplrswap', 'reidvideogui', 'dlc2vaila', 'modifylabref_cli',
        'vpython_c3d', 'compressvideo', 'markerless_2d_analysis_nvidia',
        'mphands', 'mpangles'
    ]
}

def extract_module_info(file_path):
    """Extrai informações de um módulo Python"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Parse AST para extrair funções
        try:
            tree = ast.parse(content)
            functions = [node.name for node in ast.walk(tree)
                        if isinstance(node, ast.FunctionDef) and not node.name.startswith('_')]
        except:
            functions = []

        # Extrair informações do docstring
        docstring_match = re.search(r'""".*?"""', content, re.DOTALL)
        docstring = docstring_match.group(0).strip('"""') if docstring_match else ""

        # Extrair informações básicas
        module_info = {
            'file_path': file_path,
            'module_name': Path(file_path).stem,
            'functions': functions[:20],  # Limitar para evitar arquivos muito grandes
            'docstring': docstring[:1000] + "..." if len(docstring) > 1000 else docstring,
            'has_gui': 'tkinter' in content.lower() or 'gui' in content.lower(),
            'file_size': len(content),
            'line_count': len(content.split('\n'))
        }

        # Tentar extrair versão e autor do docstring
        version_match = re.search(r'Version[:\s]+([^\n\r]+)', docstring, re.IGNORECASE)
        if version_match:
            module_info['version'] = version_match.group(1).strip()

        author_match = re.search(r'Author[:\s]+([^\n\r]+)', docstring, re.IGNORECASE)
        if author_match:
            module_info['author'] = author_match.group(1).strip()

        return module_info

    except Exception as e:
        return {
            'file_path': file_path,
            'module_name': Path(file_path).stem,
            'error': str(e),
            'functions': [],
            'docstring': '',
            'has_gui': False
        }

def get_module_category(module_name):
    """Determina a categoria de um módulo"""
    for category, modules in MODULE_CATEGORIES.items():
        if module_name in modules:
            return category
    return 'uncategorized'

def generate_html_help(module_info):
    """Gera arquivo HTML de help para um módulo"""
    category = get_module_category(module_info['module_name'])

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
            <h3>📋 Informações do Módulo</h3>
            <p><strong>Categoria:</strong> {category.title()}</p>
            <p><strong>Arquivo:</strong> {module_info['file_path']}</p>
            <p><strong>Linhas:</strong> {module_info.get('line_count', 'N/A')}</p>
            <p><strong>Tamanho:</strong> {module_info.get('file_size', 0)} caracteres</p>
            {f"<p><strong>Versão:</strong> {module_info.get('version', 'N/A')}</p>" if module_info.get('version') else ""}
            {f"<p><strong>Autor:</strong> {module_info.get('author', 'N/A')}</p>" if module_info.get('author') else ""}
            <p><strong>Interface Gráfica:</strong> {'✅ Sim' if module_info.get('has_gui') else '❌ Não'}</p>
        </div>

        {f"""
        <div class="error">
            <h3>⚠️ Erro na Análise</h3>
            <p>{module_info.get('error', 'Erro desconhecido')}</p>
        </div>
        """ if module_info.get('error') else ""}

        <div class="docstring">
            <h3>📖 Descrição</h3>
            <pre>{module_info.get('docstring', 'Sem descrição disponível')}</pre>
        </div>

        <div class="functions">
            <h3>🔧 Funções Principais</h3>
            {f"<p><strong>Total de funções encontradas:</strong> {len(module_info.get('functions', []))}</p>" if module_info.get('functions') else "<p>Nenhuma função encontrada</p>"}
            {f"<ul>{''.join(f'<li><code>{func}</code></li>' for func in module_info.get('functions', []))}</ul>" if module_info.get('functions') else ""}
        </div>

        <div class="footer">
            <p>📅 Gerado automaticamente em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
            <p>🔗 Parte do vailá - Multimodal Toolbox</p>
            <p>🌐 <a href="https://github.com/vaila-multimodaltoolbox/vaila">GitHub Repository</a></p>
        </div>
    </div>
</body>
</html>"""

    return html_content

def generate_md_help(module_info):
    """Gera arquivo MD de help para um módulo"""
    category = get_module_category(module_info['module_name'])

    md_content = f"""# {module_info['module_name']}

## 📋 Informações do Módulo

- **Categoria:** {category.title()}
- **Arquivo:** `{module_info['file_path']}`
- **Linhas:** {module_info.get('line_count', 'N/A')}
- **Tamanho:** {module_info.get('file_size', 0)} caracteres
{ f"- **Versão:** {module_info.get('version', 'N/A')}" if module_info.get('version') else "" }
{ f"- **Autor:** {module_info.get('author', 'N/A')}" if module_info.get('author') else "" }
- **Interface Gráfica:** {'✅ Sim' if module_info.get('has_gui') else '❌ Não'}

## 📖 Descrição

{module_info.get('docstring', 'Sem descrição disponível')}

## 🔧 Funções Principais

{ f"**Total de funções encontradas:** {len(module_info.get('functions', []))}" if module_info.get('functions') else "Nenhuma função encontrada" }

{ ''.join(f"- `{func}`\n" for func in module_info.get('functions', [])) if module_info.get('functions') else "" }

{ f"## ⚠️ Erro na Análise\n\n{module_info.get('error', 'Erro desconhecido')}" if module_info.get('error') else "" }

---

📅 **Gerado automaticamente em:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}  
🔗 **Parte do vailá - Multimodal Toolbox**  
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
"""

    return md_content

def main():
    """Função principal para gerar toda a documentação de help"""
    print("🔍 Iniciando análise de módulos Python do vailá...")
    print(f"📂 Diretório base: {os.getcwd()}")

    # Encontrar todos os arquivos Python no diretório vaila
    vaila_dir = Path("vaila")
    python_files = list(vaila_dir.rglob("*.py"))

    print(f"📊 Encontrados {len(python_files)} arquivos Python")

    # Analisar cada arquivo
    modules_info = []
    for py_file in python_files:
        print(f"🔍 Analisando: {py_file}")
        module_info = extract_module_info(py_file)
        modules_info.append(module_info)

    print(f"✅ Análise completa! {len(modules_info)} módulos processados")

    # Gerar arquivos de help
    help_dir = Path("vaila/help")
    help_dir.mkdir(exist_ok=True)

    generated_files = []

    for module_info in modules_info:
        category = get_module_category(module_info['module_name'])
        category_dir = help_dir / category
        category_dir.mkdir(exist_ok=True)

        # Gerar HTML
        html_content = generate_html_help(module_info)
        html_file = category_dir / f"{module_info['module_name']}.html"

        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        # Gerar MD
        md_content = generate_md_help(module_info)
        md_file = category_dir / f"{module_info['module_name']}.md"

        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(md_content)

        generated_files.append({
            'module': module_info['module_name'],
            'category': category,
            'html': str(html_file),
            'md': str(md_file)
        })

        print(f"✅ Gerado help para: {module_info['module_name']} ({category})")

    # Gerar índice geral
    index_content = generate_index(generated_files)

    with open(help_dir / "index.html", 'w', encoding='utf-8') as f:
        f.write(index_content['html'])

    with open(help_dir / "README.md", 'w', encoding='utf-8') as f:
        f.write(index_content['md'])

    print("🎉 Documentação de help gerada com sucesso!")
    print(f"📂 Arquivos criados em: {help_dir}")
    print(f"📊 Total de módulos documentados: {len(generated_files)}")
    print(f"📁 Categorias: {', '.join(set(item['category'] for item in generated_files))}")

def generate_index(generated_files):
    """Gera índice geral da documentação de help"""

    # Organizar por categoria
    categories = {}
    for item in generated_files:
        cat = item['category']
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
        <h1>📚 vailá - Documentação de Help</h1>
        <p style="text-align: center; font-size: 1.1em; color: #666; margin-bottom: 40px;">
            Documentação automática gerada para todos os módulos Python do vailá Multimodal Toolbox
        </p>

        <div style="text-align: center; margin: 30px 0;">
            <strong>Total de módulos documentados:</strong> {len(generated_files)} |
            <strong>Categorias:</strong> {len(categories)} |
            <strong>Gerado em:</strong> {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
        </div>
"""

    for category, modules in categories.items():
        html_content += f"""
        <div class="category">
            <h2>{category.title()} ({len(modules)} módulos)</h2>
            <div class="module-list">
"""

        for module in sorted(modules, key=lambda x: x['module']):
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
            <p>📧 Para dúvidas ou sugestões, entre em contato com a equipe de desenvolvimento</p>
        </div>
    </div>
</body>
</html>"""

    md_content = f"""# 📚 vailá - Documentação de Help

Documentação automática gerada para todos os módulos Python do vailá Multimodal Toolbox.

## 📊 Estatísticas Gerais

- **Total de módulos documentados:** {len(generated_files)}
- **Categorias:** {len(categories)}
- **Gerado em:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}

## 📂 Categorias de Módulos

"""

    for category, modules in categories.items():
        md_content += f"### {category.title()} ({len(modules)} módulos)\n\n"

        for module in sorted(modules, key=lambda x: x['module']):
            md_content += f"- **{module['module']}**\n"
            md_content += f"  - [📄 HTML]({category}/{module['module']}.html)\n"
            md_content += f"  - [📝 Markdown]({category}/{module['module']}.md)\n\n"

    md_content += """
## 🔗 Links Úteis

- [🌐 GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
- [📖 Documentação Principal](https://vaila.readthedocs.io/)
- [🛠️ Issues e Discussões](https://github.com/vaila-multimodaltoolbox/vaila/issues)

## 📝 Sobre Esta Documentação

Esta documentação foi gerada automaticamente através da análise dos módulos Python do vailá. Cada arquivo de help contém:

- Informações básicas do módulo (autor, versão, categoria)
- Descrição extraída do docstring
- Lista de funções principais encontradas
- Links para formatos HTML e Markdown

Para atualizar esta documentação, execute o script `generate_help_docs.py` novamente.
"""

    return {'html': html_content, 'md': md_content}

if __name__ == "__main__":
    main()
