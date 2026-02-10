# Visualização 3D de C3D (viewc3d) - vailá

Visualizador 3D avançado para arquivos C3D com escala adaptativa, detecção automática de unidades (mm/m), trilhas (trails) com ghosting, câmera Z-up para biomecânica, grade dinâmica e picking de marcadores.

**Como executar:** pelo menu da vailá (**Show C3D**) escolha **Open3D viewer** ou **PyVista viewer**; ou `uv run viewc3d.py` (Open3D).

O **viewer PyVista** tem ajuda própria: no viewer, pressione **H** para abrir a página de atalhos (`help/view3d_pyvista_help.html`). Atalhos: ← → frame, Space play/pause, S/E início/fim, R reset câmera.

---

## Arquitetura

- **VailaModel** (camada de dados): Aquisição C3D com API: `GetPointFrame`, `GetFrameNumber`, `GetPointFrequency`, `get_bounds()`, `is_z_up()`. Separação clara entre dados e visualização; permite futuro carregamento de múltiplos C3D.
- **VailaView** (camada Open3D): Cena, trilhas, grade dinâmica, câmera Z-up e picking de marcadores.

---

## Funcionalidades Principais

- Visualização adaptativa (laboratório até campo de futebol)
- Detecção automática de unidades (mm/m) com score de confiança
- Seleção interativa de marcadores (busca e filtros)
- Rótulos dos marcadores em tempo real com cores
- Grade e plano de solo adaptados ao bounding box dos dados (grade dinâmica)
- **Trilhas (trails):** histórico de movimento com segmentos mais antigos mais escuros (ghosting) e cores por velocidade
- **Câmera Z-up:** ativada automaticamente quando a amplitude em Z é maior que em Y (convenção biomecânica)
- **Picking de marcador:** teclas `` ` `` e `` ~ `` para ciclar e destacar um marcador em amarelo
- Linhas de campo (futebol) e plano de solo customizáveis
- Fallback para Matplotlib quando OpenGL não está disponível

---

## Controles de Teclado

### Navegação
- **← →** – Frame anterior/próximo
- **↑ ↓** – Avançar/voltar 60 frames
- **S / E** – Início / fim
- **Espaço ou Enter** – Play/Pause
- **F / B** – Um frame para frente/trás

### Marcadores
- **+ / =** – Aumentar tamanho
- **-** – Diminuir tamanho
- **C** – Ciclar cor dos marcadores
- **X** – Mostrar/ocultar rótulos (nomes)
- **`** (backtick) – Próximo marcador em destaque (picking)
- **~** (til) – Marcador em destaque anterior

### Visualização
- **T** – Cor de fundo
- **Y** – Cor do plano de solo
- **G** – Linhas de campo (futebol)
- **M** – Grade do solo
- **R** – Resetar câmera
- **L** – Limites de vista customizados
- **1, 2, 3, 4** – Vista frontal, direita, topo, isométrica

### Recursos avançados
- **W** – Liga/desliga trilhas (trails) com ghosting
- **J** – Carregar esqueleto (JSON)
- **D** – Medir distância entre dois marcadores
- **H** – Abrir ajuda (navegador)
- **U** – Override de unidades (mm/m)

### Captura e exportação
- **K** – Screenshot (PNG)
- **Z** – Sequência PNG
- **V** – Exportar MP4 (requer ffmpeg)
- **8** – Turntable MP4
- **Ctrl+S** – Salvar C3D editado

### Velocidade de reprodução
- **[** – Mais lento  |  **]** – Mais rápido  |  **Q** – Info de frame no console

### Mouse
- Arrastar botão esquerdo – Rotacionar
- Arrastar botão do meio/direito – Pan
- Scroll – Zoom

---

## Recursos Avançados

### Trilhas (W)
- Histórico de posições dos marcadores (ghosting)
- Segmentos mais antigos desenhados mais escuros
- Cores por magnitude de velocidade
- Atualização em tempo real durante o play

### Picking de marcador (` e ~)
- Ciclar entre marcadores para destacar um em amarelo
- Nome e índice do marcador no console

### Grade dinâmica e Z-up
- Grade e plano de solo a partir do bounding box dos dados (não mais campo fixo)
- Se amplitude em Z > Y, o vetor “up” da câmera é definido como Z (biomecânica)

### Linhas de campo personalizadas
- Carregar linhas a partir de CSV (G → carregar arquivo)
- Suporte a definição completa de campo (ref3d) ou linhas simples

### Sistema de coordenadas
- Eixos coloridos (X=vermelho, Y=verde, Z=azul)
- Escala adaptativa aos dados

---

## Inspirações e referências

- **Blender:** Grade adaptativa, vistas rápidas, FOV

---

## Arquivos de exemplo

- Linhas de campo: `tests/custom_field_lines_example.csv`
- Campo de futebol: `tests/sport_fields/soccerfield_ref3d.csv`

---

## Solução de problemas

- **OpenGL indisponível (ex.: macOS Apple Silicon):** o viewer usa fallback em Matplotlib com navegação por frame e play.
- **Marcadores não aparecem:** confira a seleção de marcadores no diálogo inicial e se o C3D tem dados válidos no intervalo de frames.

---

## Dependências

- open3d, numpy, pandas, matplotlib, ezc3d, rich
- ffmpeg (opcional, para exportar MP4)

Para mais atalhos e detalhes, use a tecla **H** no viewer para abrir a ajuda no navegador.
