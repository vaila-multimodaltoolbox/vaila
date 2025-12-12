#!/bin/bash
# Script rápido para instalar dependências do Cairo necessárias para pycairo

echo "Instalando dependências do Cairo para pycairo..."
sudo apt update
sudo apt install -y libcairo2-dev pkg-config python3-dev build-essential

echo ""
echo "Dependências instaladas! Agora você pode executar:"
echo "  uv run vaila.py"

