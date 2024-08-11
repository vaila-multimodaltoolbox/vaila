# README.md (English-US)

## vailá - multimodal toolbox - conda environments

## About

This repository contains YAML files to set up the Conda environment required for the **vailá - multimodal toolbox** project. There are separate files for different operating systems: Linux (Ubuntu), macOS, and Windows 11.

## Repository Structure

- `vaila_linux.yaml`: Conda environment configuration for Linux (Ubuntu 22.04).
- `vaila_macos.yaml`: Conda environment configuration for macOS.
- `vaila_win.yaml`: Conda environment configuration for Windows 11.

## Installation Instructions

### Linux (Ubuntu)

1. **Install Anaconda**

   Follow the instructions on the official site to install Anaconda:
   - [Anaconda](https://www.anaconda.com/products/individual)

2. **Open a terminal and navigate to the directory where the `vaila_linux.yaml` file is located.**

3. **Create the environment**

   ```bash
   conda env create -f vaila_linux.yaml

   ```

4. **Activate the environment**

   ```bash
   conda activate vaila

   ```

### macOS

1. **Install Anaconda**

   Follow the instructions on the official site to install Anaconda:
   - [Anaconda](https://www.anaconda.com/products/individual)

2. **Open a terminal (zsh by default) and navigate to the directory where the `vaila_macos.yaml` file is located.**

3. **Create the environment**

   ```zsh
   conda env create -f vaila_macos.yaml
   ```

4. **Activate the environment**

   ```zsh
   conda activate vaila
   ```

### Windows 11

1. **Install Anaconda**

   Follow the instructions on the official site to install Anaconda:
   - [Anaconda](https://www.anaconda.com/products/individual)

2. **Open Anaconda Prompt and navigate to the directory where the `vaila_win.yaml` file is located.**

3. **Create the environment**

   ```bash
   conda env create -f vaila_win.yaml
   ```

4. **Activate the environment**

   ```bash
   conda activate vaila
   ```

## Update Environment

To update the existing environment based on changes in the YAML files, run:

```bash
conda env update -f <filename>.yaml
```

Replace `<filename>` with the appropriate YAML file name (`vaila_linux.yaml`, `vaila_macos.yaml`, or `vaila_win.yaml`).

---

### README.md (Português-BR)

# vailá - multimodal toolbox - ambientes conda

## Sobre

Este repositório contém arquivos YAML para configurar o ambiente Conda necessário para o projeto **vailá - multimodal toolbox**. Existem arquivos separados para diferentes sistemas operacionais: Linux (Ubuntu), macOS e Windows 11.

## Estrutura do Repositório

- `vaila_linux.yaml`: Configuração do ambiente Conda para Linux (Ubuntu).
- `vaila_macos.yaml`: Configuração do ambiente Conda para macOS.
- `vaila_win11.yaml`: Configuração do ambiente Conda para Windows 11.

## Instruções de Instalação

### Linux (Ubuntu)

1. **Instalar Anaconda**

   Siga as instruções no site oficial para instalar o Anaconda:
   - [Anaconda](https://www.anaconda.com/products/individual)

2. **Abra um terminal e navegue até o diretório onde o arquivo `vaila_linux.yaml` está localizado.**

3. **Criar o ambiente**

   ```bash
   conda env create -f vaila_linux.yaml

   ```

4. **Ativar o ambiente**

   ```bash
   conda activate vaila
   ```

### macOS

1. **Instalar Anaconda**

   Siga as instruções no site oficial para instalar o Anaconda:
   - [Anaconda](https://www.anaconda.com/products/individual)

2. **Abra um terminal (zsh por padrão) e navegue até o diretório onde o arquivo `vaila_macos.yaml` está localizado.**

3. **Criar o ambiente**

   ```zsh
   conda env create -f vaila_macos.yaml
   ```

4. **Ativar o ambiente**

   ```zsh
   conda activate vaila
   ```

### Windows 11

1. **Instalar Anaconda**

   Siga as instruções no site oficial para instalar o Anaconda:
   - [Anaconda](https://www.anaconda.com/products/individual)

2. **Abra o Anaconda Prompt e navegue até o diretório onde o arquivo `vaila_win.yaml` está localizado.**

3. **Criar o ambiente**

   ```bash
   conda env create -f vaila_win.yaml
   ```

4. **Ativar o ambiente**

   ```bash
   conda activate vaila
   ```

## Atualização do Ambiente

Para atualizar o ambiente existente com base nas mudanças nos arquivos YAML, execute:

```bash
conda env update -f <nome_do_arquivo>.yaml
```

Substitua `<nome_do_arquivo>` pelo nome do arquivo YAML correspondente ao seu sistema operacional (`vaila_linux.yaml`, `vaila_macos.yaml` ou `vaila_win.yaml`).
