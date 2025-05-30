# Guia do Usuário - Pixel Coordinate Tool (getpixelvideo.py)

## Introdução

A ferramenta Pixel Coordinate Tool (getpixelvideo.py) permite marcar e salvar coordenadas de pixels em quadros de vídeo. Desenvolvida pelo Prof. Dr. Paulo R. P. Santiago, esta ferramenta oferece funcionalidades como zoom para anotações precisas, redimensionamento dinâmico da janela, navegação entre quadros e capacidade de salvar resultados em formato CSV.

## Requisitos

- Python 3.12 ou superior
- Bibliotecas: pygame, cv2 (OpenCV), pandas, numpy, tkinter

## Começando

1. Execute o script usando Python: `python getpixelvideo.py`
2. Selecione um arquivo de vídeo quando solicitado
3. Escolha se deseja carregar pontos existentes de um arquivo salvo anteriormente

## Interface

A interface da ferramenta consiste em:
- Área de visualização do vídeo (parte superior)
- Painel de controle (parte inferior) com:
  - Informações do quadro atual
  - Slider para navegação entre quadros
  - Botões para funções principais (Carregar, Salvar, Ajuda, 1 Line, Persistência, Sequencial)

## Modos de Marcação

### Modo Normal (padrão)
- Cada clique seleciona e atualiza o marcador atualmente selecionado
- Navegue entre marcadores usando TAB
- Cada marcador mantém seu ID em todos os quadros
- Ative com: modo padrão ao iniciar

### Modo 1 Line (tecla C)
- Cria uma sequência de pontos conectados em um único quadro
- Cada clique adiciona um novo marcador em ordem sequencial
- Útil para traçar caminhos ou contornos
- Ative com: tecla C

### Modo Sequencial (tecla S)
- Cada clique cria um novo marcador com IDs incrementais
- Não é necessário selecionar marcadores primeiro
- Disponível apenas no modo Normal
- Ative com: tecla S (apenas no modo Normal)

## Comandos do Teclado

### Navegação de Vídeo
- **Espaço**: Reproduzir/Pausar
- **Seta Direita**: Próximo quadro (quando pausado)
- **Seta Esquerda**: Quadro anterior (quando pausado)
- **Seta Para Cima**: Avançar rápido (quando pausado)
- **Seta Para Baixo**: Retroceder (quando pausado)

### Zoom e Navegação
- **+**: Aumentar zoom
- **-**: Diminuir zoom
- **Clique do Meio + Arrastar**: Mover a visualização (pan)

### Marcadores
- **Clique Esquerdo**: Adicionar/atualizar marcador
- **Clique Direito**: Remover último marcador
- **TAB**: Próximo marcador no quadro atual
- **SHIFT+TAB**: Marcador anterior no quadro atual
- **DELETE**: Excluir marcador selecionado
- **A**: Adicionar novo marcador vazio ao arquivo
- **R** ou **D**: Remover marcador selecionado

### Modos
- **C**: Alternar modo "1 Line"
- **S** ou **O**: Alternar modo Sequencial (apenas no modo Normal)
- **P**: Alternar modo de Persistência
- **1**: Diminuir quadros de persistência (quando persistência ativada)
- **2**: Aumentar quadros de persistência (quando persistência ativada)
- **3**: Alternar entre persistência completa e limitada

### Outros
- **ESC**: Salvar e sair

## Modo de Persistência

O modo de persistência mostra marcadores de quadros anteriores, criando um "rastro" visual:
- **P**: Ativa/desativa a persistência
- **1**: Diminui o número de quadros exibidos
- **2**: Aumenta o número de quadros exibidos
- **3**: Alterna entre modos de persistência (desativado → completo → limitado)

## Salvando e Carregando

### Salvar Coordenadas
- Pressione **ESC** para salvar e sair
- Clique no botão **Salvar** para salvar sem sair
- Arquivos são salvos como CSV no mesmo diretório do vídeo
- Diferentes modos salvam em arquivos diferentes:
  - Modo Normal: `nome_do_video_markers.csv`
  - Modo 1 Line: `nome_do_video_markers_1_line.csv`
  - Modo Sequencial: `nome_do_video_markers_sequential.csv`

### Carregar Coordenadas
- Selecione "Sim" quando perguntado ao iniciar
- Ou clique no botão **Carregar** a qualquer momento

## Dicas

1. Use o modo Sequencial quando quiser criar múltiplos marcadores sem se preocupar com a seleção
2. Use o modo 1 Line para traçar contornos ou caminhos em um único quadro
3. Backups automáticos são criados com timestamp para evitar perda de dados
4. Utilize o zoom para maior precisão ao marcar coordenadas
5. A tecla **A** é útil para adicionar marcadores vazios que podem ser preenchidos posteriormente

## Solução de Problemas

- Se a janela não mostrar o vídeo: verifique se o arquivo de vídeo está no formato suportado
- Se os marcadores não aparecerem ao carregar: verifique se o arquivo CSV está no formato correto
- Se a performance estiver lenta: reduza o tamanho da janela ou desative o modo de persistência
