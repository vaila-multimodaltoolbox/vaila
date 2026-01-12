# vailá - Versatile Anarcho Integrated Multimodal Toolbox Help

## English Version

### Overview
vailá is an open-source multimodal toolbox for human movement analysis. It integrates data from various sources – including IMU, MoCap, markerless tracking, face mesh detection, GNSS/GPS, EMG, and more – enabling advanced and customizable analysis.

### Key Features
- **Data Integration:** Supports multiple data types (IMU, MoCap, markerless, face mesh, GNSS, EMG).
- **Data Processing & Analysis:** Feature extraction, advanced analysis, and 2D/3D visualization.
- **Machine Learning:** Modules for training, validation, and prediction using ML models.
- **File Management:** Organization, renaming, copying, and file movement.
- **Video Processing:** Extraction of frames, compression (H.264 and H.265), and video trimming.

### Installation Instructions

#### ⚡ Powered by *uv* (Recommended)
*vailá* now uses **[uv](https://github.com/astral-sh/uv)**, an extremely fast Python package installer. **uv is the recommended installation method** for all platforms due to its **10-100x faster installation** and **faster execution times** compared to Conda.

#### Prerequisites
- **uv:** Will be automatically installed by the installation scripts, or install manually from [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv)
- **FFmpeg:** Required for video processing functionalities (installed automatically on Windows)

#### Steps
1. **Clone the repository:**
   ```bash
   git clone https://github.com/vaila-multimodaltoolbox/vaila.git
   cd vaila
   ```

2. **Set up the environment:**
   - **Windows (Recommended):** Run `.\install_vaila_win.ps1`
     The script will prompt you to choose between:
     - **uv** (recommended - modern, fast)
     - **Conda** (legacy - for compatibility)
   
   - **Linux and macOS (Using uv):**
     ```bash
     # Install uv
     curl -LsSf https://astral.sh/uv/install.sh | sh
     
     # Sync dependencies
     uv sync
     ```
   
   - **Legacy Conda Method (slower):**
     - **Linux:** Run `./install_vaila_linux.sh`
     - **macOS:** Run `./install_vaila_mac.sh`
     - **Windows:** Run `.\install_vaila_win.ps1`

3. **Run vailá:**
   **Using uv (Recommended):**
   ```bash
   uv run vaila.py
   ```
   
   **Using Conda (Legacy):**
   ```bash
   conda activate vaila
   python vaila.py
   ```

### GUI Button Documentation
All buttons in the vailá GUI are documented. See the [Button Documentation](vaila_buttons/README.md) for complete details. (**Note:** Check if this directory exists or update link)

- **[Markerless 2D Analysis](vaila_buttons/markerless-2d-button.html)** - Advanced pose estimation
- **[All Button Documentation](vaila_buttons/README.md)** - Complete list of all GUI buttons

### Script Help Documentation
Comprehensive documentation for all Python scripts and modules in vailá:
- **[Script Help Index](../vaila/help/index.html)** - Complete documentation for all Python modules and scripts

### Video Processing Tools
- **[DrawBoxe](../vaila/help/drawboxe.html)** - Draw boxes and polygons on videos with frame interval support

### Modules and Tools
The vailá toolbox comprises the following modules:
- **IMU Analysis**
- **MoCap Analysis** (Cluster, Full Body)
- **Markerless Analysis** (2D/3D)
- **Face Mesh Analysis** - Face and iris landmark detection (478 landmarks)
- **Force Plate Analysis**
- **GNSS/GPS Analysis**
- **EEG/EMG Analysis**
- **ML Walkway:**
  - Model Training
  - Model Validation
  - Prediction with Pre-trained Models
- **File Management**
- **Video Processing:**
  - DrawBoxe - Draw boxes and polygons on videos
- **Visualization**

### How to Use
After setting up the environment, run vailá using:

**Using uv (Recommended):**
```bash
uv run vaila.py
```

**Using Conda (Legacy):**
```bash
conda activate vaila
python vaila.py
```

The graphical interface allows you to select the desired module. For example, when selecting "ML Walkway," a window with options for model training, validation, or prediction will open.

### Contributing
Contributions are welcome! If you encounter issues or have suggestions, please submit a pull request or open an issue on GitHub.

### License
This project is licensed under the **GNU Lesser General Public License v3.0**. See the `LICENSE` file for more details.

### How to Cite
If vailá is useful for your research, please cite:
```bibtex
@misc{vaila2024,
  title={vailá - Versatile Anarcho Integrated Liberation Ánalysis in Multimodal Toolbox},
  author={Paulo R. P. Santiago and Abel G. Chinaglia and others},
  year={2024},
  url={https://github.com/vaila-multimodaltoolbox/vaila}
}
```

---

## Versão em Português

### Visão Geral
vailá é uma ferramenta multimodal de código aberto para análise do movimento humano. Ela integra dados de diversas fontes – como IMU, MoCap, rastreamento markerless, detecção de face mesh, GNSS/GPS, EMG e mais – permitindo uma análise avançada e customizável.

### Principais Funcionalidades
- **Integração de Dados:** Suporte a múltiplos tipos de dados (IMU, MoCap, markerless, face mesh, GNSS, EMG).
- **Processamento & Análise de Dados:** Extração de características, análise avançada e visualização em 2D/3D.
- **Machine Learning:** Módulos para treinamento, validação e predição utilizando modelos de ML.
- **Gerenciamento de Arquivos:** Organização, renomeação, cópia e movimentação de arquivos.
- **Processamento de Vídeo:** Extração de frames, compressão (H.264 e H.265) e corte de vídeos.

### Instruções de Instalação

#### ⚡ Powered by *uv* (Recomendado)
*vailá* agora usa **[uv](https://github.com/astral-sh/uv)**, um instalador de pacotes Python extremamente rápido. **uv é o método de instalação recomendado** para todas as plataformas devido à sua **instalação 10-100x mais rápida** e **tempos de execução mais rápidos** em comparação com Conda.

#### Pré-requisitos
- **uv:** Será instalado automaticamente pelos scripts de instalação, ou instale manualmente em [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv)
- **FFmpeg:** Necessário para funcionalidades de processamento de vídeo (instalado automaticamente no Windows)

#### Passos
1. **Clone o repositório:**
   ```bash
   git clone https://github.com/vaila-multimodaltoolbox/vaila.git
   cd vaila
   ```

2. **Configurar o ambiente:**
   - **Windows (Recomendado):** Execute `.\install_vaila_win.ps1`
     O script solicitará que você escolha entre:
     - **uv** (recomendado - moderno, rápido)
     - **Conda** (legado - para compatibilidade)
   
   - **Linux e macOS (Usando uv):**
     ```bash
     # Instalar uv
     curl -LsSf https://astral.sh/uv/install.sh | sh
     
     # Sincronizar dependências
     uv sync
     ```
   
   - **Método Conda Legacy (mais lento):**
     - **Linux:** Execute `./install_vaila_linux.sh`
     - **macOS:** Execute `./install_vaila_mac.sh`
     - **Windows:** Execute `.\install_vaila_win.ps1`

3. **Inicie o vailá:**
   **Usando uv (Recomendado):**
   ```bash
   uv run vaila.py
   ```
   
   **Usando Conda (Legacy):**
   ```bash
   conda activate vaila
   python vaila.py
   ```

### Documentação de Help dos Scripts
Documentação completa para todos os scripts e módulos Python em vailá:
- **[Índice de Help dos Scripts](../vaila/help/index.html)** - Documentação completa para todos os módulos e scripts Python

### Módulos e Ferramentas
O toolbox vailá é composto pelos seguintes módulos:
- **Análise IMU**
- **Análise MoCap** (Cluster, Full Body)
- **Análise Markerless** (2D/3D)
- **Análise Face Mesh** - Detecção de landmarks faciais e íris (478 landmarks)
- **Análise de Force Plate**
- **Análise GNSS/GPS**
- **Análise EEG/EMG**
- **ML Walkway:**
  - Treinamento de Modelos
  - Validação de Modelos
  - Predição com Modelos Pré-treinados
- **Gerenciamento de Arquivos**
- **Processamento de Vídeo:**
  - DrawBoxe - Desenhar caixas e polígonos em vídeos
- **Visualização**

### Ferramentas de Processamento de Vídeo
- **[DrawBoxe](../vaila/help/drawboxe.html)** - Desenhar caixas e polígonos em vídeos com suporte a intervalos de frames

### Como Utilizar
Após configurar o ambiente, inicie o vailá com o comando:

**Usando uv (Recomendado):**
```bash
uv run vaila.py
```

**Usando Conda (Legacy):**
```bash
conda activate vaila
python vaila.py
```

A interface gráfica permitirá que você selecione o módulo desejado. Por exemplo, ao selecionar "ML Walkway", uma janela com opções para treinamento, validação ou predição de modelos será aberta.

### Contribuição
Suas contribuições são bem-vindas! Caso encontre problemas ou tenha sugestões, por favor, envie um pull request ou abra uma issue no GitHub.

### Licença
Este projeto está licenciado sob a **GNU Lesser General Public License v3.0**. Consulte o arquivo `LICENSE` para mais detalhes.

### Como Citar
Se o vailá for útil na sua pesquisa, por favor, cite:
```bibtex
@misc{vaila2024,
  title={vailá - Versatile Anarcho Integrated Liberation Ánalysis in Multimodal Toolbox},
  author={Paulo R. P. Santiago e Abel G. Chinaglia e outros},
  year={2024},
  url={https://github.com/vaila-multimodaltoolbox/vaila}
}
```
