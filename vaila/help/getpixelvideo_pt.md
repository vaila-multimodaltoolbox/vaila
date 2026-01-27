# Guia do Usuário - Pixel Coordinate Tool (getpixelvideo.py)

## Introdução

A ferramenta Pixel Coordinate Tool (getpixelvideo.py) é uma ferramenta abrangente de anotação de vídeo que permite marcar e salvar coordenadas de pixels em quadros de vídeo. Desenvolvida pelo Prof. Dr. Paulo R. P. Santiago, esta ferramenta oferece recursos avançados incluindo zoom para anotações precisas, redimensionamento dinâmico da janela, navegação entre quadros, suporte a múltiplos formatos CSV e capacidades avançadas de visualização de dados.

**Version:** 0.3.0  
**Data:** Janeiro de 2026  
**Autores:** Prof. Dr. Paulo R. P. Santiago, Rafael L. M. Monteiro  
**Projeto:** vailá - Multimodal Toolbox

## Principais Recursos

- **Suporte Multi-formato:** Carregar e visualizar formatos MediaPipe, YOLO tracking e vailá padrão
- **Visualização Avançada:** Figuras de palitos para MediaPipe, caixas delimitadoras para YOLO tracking
- **Marcação Flexível:** Múltiplos modos de marcadores para diferentes necessidades de anotação
- **Modo Labeling:** Criar anotações de caixas delimitadoras para datasets de Machine Learning
- **Exportação de Datasets:** Exportar datasets estruturados (train/val/test) com imagens e anotações JSON
- **Zoom & Navegação:** Capacidades completas de zoom com navegação quadro a quadro
- **Modo Persistência:** Visualizar trilhas de marcadores através de múltiplos quadros
- **Auto-detecção:** Detectar automaticamente formato CSV ou seleção manual
- **Documentação Abrangente:** Arquivo de ajuda HTML com instruções detalhadas

## Requisitos

- **Python 3.12+**
- **OpenCV (cv2):** Processamento de vídeo
- **Pygame:** Interface gráfica e visualização
- **Pandas:** Manipulação de dados CSV
- **NumPy:** Operações numéricas
- **Tkinter:** Diálogos de arquivo (geralmente incluído com Python)

### Instalação

```bash
pip install opencv-python pygame pandas numpy
```

## Começando

1. **Execute o script:** `python vaila/getpixelvideo.py`
2. **Selecione arquivo de vídeo:** Escolha o vídeo para processar
3. **Carregue dados existentes:** Use o botão 'Load' na interface para carregar keypoints (opcional)
4. **Selecione formato:** Se carregando dados, escolha o formato CSV:
   - **Auto-detectar (recomendado):** Detecta automaticamente o formato
   - **Formato MediaPipe:** Para dados de landmarks com visualização de stick figure
   - **Formato YOLO tracking:** Para dados de tracking com visualização de bounding box
   - **Formato vailá padrão:** Para dados de coordenadas padrão
5. **Navegue e anote:** Use a interface para navegar, fazer zoom e editar marcadores
6. **Salve resultados:** Salve os dados anotados em formato CSV

## Interface

A interface da ferramenta consiste em:

- **Área de visualização do vídeo** (seção superior) com capacidades de zoom e pan
- **Painel de controle** (seção inferior) com:
  - Informações do quadro atual
  - Slider para navegar entre quadros
  - Botões para functions main (Carregar, Salvar, Ajuda, 1 Line, Persistência, Sequencial)
  - Controles de visualização específicos do formato

## Formatos de Arquivo Suportados

### Formato MediaPipe

Usado para dados de estimação de pose e detecção de landmarks.

**Formato:** `frame, landmark_0_x, landmark_0_y, landmark_0_z, landmark_1_x, landmark_1_y, landmark_1_z, ...`

**Exemplo:**

```csv
frame,landmark_0_x,landmark_0_y,landmark_0_z,landmark_1_x,landmark_1_y,landmark_1_z
0,100.5,200.3,0.0,150.2,250.1,0.0
1,105.2,205.1,0.0,155.3,255.2,0.0
```

**Visualização:** Figuras de palitos com landmarks conectados por lines

### Formato YOLO Tracking

Usado para dados de tracking e detecção de objetos.

**Formato:** `Frame, Tracker ID, Label, X_min, Y_min, X_max, Y_max, Confidence, Color_R, Color_G, Color_B`

**Exemplo:**

```csv
Frame,Tracker ID,Label,X_min,Y_min,X_max,Y_max,Confidence,Color_R,Color_G,Color_B
0,1,person,100,200,200,300,0.9,255,0,0
0,2,person,300,400,400,500,0.8,0,255,0
```

**Visualização:** Caixas delimitadoras com labels e IDs de tracker

### Formato vailá Padrão

Formato de coordenadas padrão usado pela toolbox vailá.

**Formato:** `frame, p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, ...`

**Exemplo:**

```csv
frame,p1_x,p1_y,p2_x,p2_y
0,100.5,200.3,150.2,250.1
1,105.2,205.1,155.3,255.2
```

**Visualização:** Marcadores pontuais com IDs

## Modos de Marcação

### Modo Normal (padrão)

- Cada clique seleciona e atualiza o marcador atualmente selecionado
- Navegue entre marcadores usando TAB
- Cada marcador mantém seu ID em todos os quadros
- **Caso de uso:** Rastrear pontos específicos através dos quadros
- **Ativação:** Modo padrão ao iniciar

### Modo 1 Line (tecla C)

- Cria pontos em sequência em um único quadro
- Cada clique adiciona um novo marcador em ordem sequencial
- **Caso de uso:** Traçar contornos, caminhos ou outlines
- **Ativação:** Pressione tecla C para alternar
- **Comportamento:** Cada clique cria um novo marcador sequencial

### Modo Sequencial (tecla S)

- Cada clique cria um novo marcador com IDs incrementais
- Não é necessário selecionar marcadores primeiro
- Disponível apenas no modo Normal
- **Caso de uso:** Anotação rápida de múltiplos pontos
- **Ativação:** Pressione tecla S para alternar
- **Comportamento:** Incremento automático de ID para cada novo marcador

### Modo de Labeling (tecla L)

- Desenha caixas delimitadoras (bounding boxes) em frames de vídeo
- Cria datasets para treinamento de detecção de objetos (formato YOLO/COCO)
- **Caso de uso:** Criação de datasets para Machine Learning
- **Ativação:** Pressione tecla L ou clique no botão "Labeling"
- **Comportamento:**
  - Clique e arraste para desenhar caixas delimitadoras
  - Pressione Z ou Clique com Botão Direito para remover a última caixa no frame atual
  - Pressione N para renomear o label do objeto atual
  - Pressione F5 para salvar o projeto de labeling (JSON)
  - Pressione F6 para carregar um projeto de labeling (JSON)
  - Salvar exporta dataset estruturado (train/val/test)
- **Exportação:** Gera estrutura de pastas com imagens e anotações JSON

## Comandos do Teclado

### Navegação de Vídeo

| Tecla               | Ação                             |
| ------------------- | -------------------------------- |
| **Espaço**          | Reproduzir/Pausar                |
| **→**               | Próximo quadro (quando pausado)  |
| **←**               | Quadro anterior (quando pausado) |
| **↑**               | Avançar rápido (quando pausado)  |
| **↓**               | Retroceder (quando pausado)      |
| **Arrastar Slider** | Pular para quadro específico     |

### Zoom e Pan

| Tecla              | Ação                |
| ------------------ | ------------------- |
| **+**              | Aumentar zoom       |
| **-**              | Diminuir zoom       |
| **Roda do Mouse**  | Zoom in/out         |
| **Clique do Meio** | Habilitar pan/mover |

### Gerenciamento de Marcadores

| Tecla               | Ação                                     |
| ------------------- | ---------------------------------------- |
| **Clique Esquerdo** | Adicionar/atualizar marcador             |
| **Clique Direito**  | Remover último marcador                  |
| **TAB**             | Próximo marcador no quadro atual         |
| **SHIFT+TAB**       | Marcador anterior no quadro atual        |
| **DELETE**          | Excluir marcador selecionado             |
| **A**               | Adicionar novo marcador vazio ao arquivo |
| **R**               | Remover último marcador do arquivo       |

### Controles de Modo

| Tecla           | Ação                                                       |
| --------------- | ---------------------------------------------------------- |
| **C**           | Alternar modo "1 Line"                                     |
| **S**           | Alternar modo Sequencial (apenas no modo Normal)           |
| **P**           | Alternar modo Persistência                                 |
| **L**           | Alternar modo Labeling (Bounding Boxes)                    |
| **Z / R-Click** | Remover última caixa delimitadora (modo Labeling)          |
| **N**           | Renomear label do objeto (Apenas Modo Labeling)            |
| **F5**          | Salvar Projeto de Labeling (JSON) (Apenas Modo Labeling)   |
| **F6**          | Carregar Projeto de Labeling (JSON) (Apenas Modo Labeling) |
| **1**           | Diminuir quadros de persistência                           |
| **2**           | Aumentar quadros de persistência                           |
| **3**           | Alternar persistência completa                             |

### Operações de Arquivo

| Tecla | Ação                                  |
| ----- | ------------------------------------- |
| **S** | Salvar marcadores atuais              |
| **B** | Fazer backup dos dados atuais         |
| **L** | Recarregar coordenadas do arquivo     |
| **H** | Mostrar diálogo de ajuda              |
| **D** | Abrir documentação completa (no help) |

### Outros

| Tecla   | Ação          |
| ------- | ------------- |
| **ESC** | Salvar e sair |

## Visualização de Dados

### Visualização MediaPipe

Ao carregar dados MediaPipe, a ferramenta exibe figuras de palitos:

- **Landmarks:** Pontos vermelhos para cada ponto detectado
- **Conexões:** Linhas verdes conectando landmarks relacionados
- **Estrutura de Pose:** Pose corporal completa com cabeça, braços, torso e pernas

### Visualização YOLO Tracking

Ao carregar dados YOLO tracking, a ferramenta exibe caixas delimitadoras:

- **Caixas Delimitadoras:** Retângulos coloridos ao redor de objetos detectados
- **Labels:** Classe do objeto e ID do tracker exibidos
- **Cores:** Cores únicas para diferentes IDs de tracker
- **Confiança:** Valores de confiança de detecção mostrados

### Visualização vailá Padrão

Visualização padrão de marcadores pontuais:

- **Marcadores:** Círculos verdes para cada ponto
- **Números:** IDs dos marcadores exibidos ao lado dos pontos
- **Seleção:** Destaque laranja para marcador selecionado

## Modo de Persistência

O modo de persistência mostra marcadores de quadros anteriores, criando um "rastro" visual:

- **P:** Habilita/desabilita persistência
- **1:** Diminui o número de quadros exibidos
- **2:** Aumenta o número de quadros exibidos
- **3:** Alterna entre modos de persistência (desabilitado → completo → limitado)

**Recursos:**

- Trilhas desvanecidas mostram movimento dos marcadores
- Número configurável de quadros para exibir
- Feedback visual para trajetórias dos marcadores

## Modo de Labeling (Bounding Boxes)

O modo de labeling permite criar anotações de caixas delimitadoras para treinamento de modelos de detecção de objetos.

**Como usar:**

1. **Ativar:** Pressione a tecla **L** ou clique no botão "Labeling" (ficha verde quando ativo)
2. **Desenhar caixas:** Clique e arraste no vídeo para desenhar caixas delimitadoras
3. **Editar:** Pressione **Z** para remover a última caixa no frame atual
4. **Exportar:** Clique no botão "Save" ou pressione **ESC** para exportar o dataset

**Exportação do Dataset:**
Ao salvar no modo labeling, o sistema cria uma estrutura de dataset organizada:

- **train/** (70% dos frames anotados)
  - `images/` - Imagens dos frames
  - `labels/` - Anotações JSON
- **val/** (20% dos frames anotados)
  - `images/` - Imagens dos frames
  - `labels/` - Anotações JSON
- **test/** (10% dos frames anotados)
  - `images/` - Imagens dos frames
  - `labels/` - Anotações JSON

**Formato JSON:**
Cada arquivo JSON contém:

- `image`: Nome do arquivo de imagem
- `width`: Largura do vídeo original
- `height`: Altura do vídeo original
- `annotations`: Array de caixas delimitadoras com campos:
  - `x`: Coordenada X (pixel)
  - `y`: Coordenada Y (pixel)
  - `w`: Largura (pixel)
  - `h`: Altura (pixel)
  - `label`: Classe do objeto (padrão: "object")

**Recursos:**

- Caixas delimitadoras são salvas por frame
- Visualização em tempo real durante o desenho
- Divisão automática train/val/test (70/20/10)
- Dataset pronto para uso em frameworks de ML (YOLO, COCO, etc.)

## Salvando e Carregando

### Opções de Salvamento

#### Salvamento Padrão (tecla S)

- **Formato:** `frame, p1_x, p1_y, p2_x, p2_y, ...`
- **Arquivo:** `{nome_do_video}_markers.csv`
- **Localização:** Mesmo diretório do arquivo de vídeo

#### Salvamento 1 Line

- **Formato:** `frame, p1_x, p1_y, p2_x, p2_y, ...`
- **Arquivo:** `{nome_do_video}_markers_sequential.csv`
- **Uso:** Para traçado de caminhos e dados de contorno

#### Salvamento Sequencial

- **Formato:** `frame, p1_x, p1_y, p2_x, p2_y, ...`
- **Arquivo:** `{nome_do_video}_markers_sequential.csv`
- **Uso:** Para anotações de múltiplos pontos

#### Salvamento Labeling (Modo de Caixas Delimitadoras)

- **Formato:** Dataset estruturado com imagens e anotações JSON
- **Diretório:** `{nome_do_video}_dataset/`
- **Estrutura:** train/val/test com images/ e labels/
- **Uso:** Para criação de datasets de Machine Learning (detecção de objetos)
- **Ativação:** Salvar quando o modo Labeling estiver ativo

### Carregando Coordenadas

- Clique no botão **Carregar** a qualquer momento
- **Auto-detecção:** Detecta automaticamente formato CSV
- **Seleção manual:** Escolha formato manualmente se necessário

## Recursos Avançados

### Auto-detecção

A ferramenta detecta automaticamente formato CSV baseado na estrutura das colunas:

- **MediaPipe:** Detecta 'landmark' nos nomes das colunas
- **YOLO:** Detecta colunas 'Frame', 'Tracker ID', 'X_min'
- **vailá:** Detecta padrão 'frame' e 'p' nas colunas

### Backup e Recuperação

Sistema de backup integrado para segurança dos dados:

- **Backup:** Pressione B para criar backup
- **Recarregar:** Pressione L para recarregar do arquivo
- **Auto-backup:** Backups automáticos antes de operações importantes

### Acesso à Documentação

- **Ajuda Rápida:** Pressione H para ajuda no aplicativo
- **Documentação Completa:** Pressione D no diálogo de ajuda para documentação HTML completa
- **Documentação HTML:** Localizada em `vaila/help/getpixelvideo.html`

## Dicas e Melhores Práticas

1. **Use o modo Sequencial** quando quiser criar múltiplos marcadores sem se preocupar com seleção
2. **Use o modo 1 Line** para traçar contornos ou caminhos em um único quadro
3. **Backups automáticos** são criados com timestamps para evitar perda de dados
4. **Use zoom** para maior precisão ao marcar coordenadas
5. **A tecla A** é útil para adicionar marcadores vazios que podem ser preenchidos posteriormente
6. **O modo persistência** é ótimo para visualizar padrões de movimento
7. **Auto-detecção** funciona melhor com arquivos CSV formatados corretamente
8. **Faça backup regularmente** usando a tecla B para evitar perda de dados

## Solução de Problemas

### Problemas Comuns

#### Vídeo Não Carrega

- Verifique se o arquivo de vídeo está corrompido
- Certifique-se de que o formato de vídeo é suportado (MP4, AVI, MOV, MKV)
- Verifique se o caminho do arquivo não contém caracteres especiais

#### Formato CSV Não Detectado

- Verifique se a estrutura do arquivo CSV corresponde ao formato esperado
- Use seleção manual de formato se a auto-detecção falhar
- Verifique se o arquivo CSV não está corrompido

#### Problemas de Performance

- Reduza a resolução do vídeo para melhor performance
- Feche outros aplicativos para liberar memória
- Use níveis menores de zoom para vídeos grandes

#### Problemas de Visualização

- Certifique-se de que os dados CSV estão no formato correto
- Verifique se os valores de coordenadas estão dentro das dimensões do vídeo
- Verifique as conexões de landmarks para dados MediaPipe

### Mensagens de Erro

| Erro                        | Solução                                             |
| --------------------------- | --------------------------------------------------- |
| "Error opening video file"  | Verifique formato do vídeo e integridade do arquivo |
| "No keypoint file selected" | Selecione um arquivo CSV válido ou comece do zero   |
| "Unknown format"            | Use seleção manual de formato                       |
| "Error loading coordinates" | Verifique formato e estrutura do arquivo CSV        |

## Suporte e Documentação

- **Ajuda no Aplicativo:** Pressione H para ajuda rápida
- **Documentação Completa:** Pressione D no diálogo de ajuda para documentação HTML completa
- **Documentação HTML:** `vaila/help/getpixelvideo.html`
- **Repositório do Projeto:** https://github.com/paulopreto/vaila-multimodaltoolbox

## Histórico de Versões

### Versão 0.3.0 (Janeiro de 2026)

- Adicionado Modo de Labeling (Bounding Boxes) para criação de datasets de Machine Learning
- Implementada exportação de datasets estruturados (train/val/test)
- Adicionado suporte para anotações JSON com formato customizado
- Criada função de exportação automática com divisão 70/20/10
- Adicionado botão "Labeling" na interface
- Melhorado help dialog com instruções detalhadas do modo labeling
- **Melhorias:**
  - Adicionado Salvar/Carregar Projeto (F5/F6) para sessões de labeling (formato JSON)
  - Adicionada habilidade de Renomear labels de objetos (tecla N)
  - Adicionado Scroll no Diálogo de Ajuda
  - Compatibilidade Linux melhorada (movido Tkinter para CLI)

### Versão 0.0.8 (27 de Julho de 2025)

- Adicionado suporte para múltiplos formatos CSV (MediaPipe, YOLO tracking, vailá padrão)
- Implementada auto-detecção de formato CSV
- Adicionada visualização para figuras de palitos MediaPipe e caixas delimitadoras YOLO
- Criada documentação HTML abrangente
- Adicionado acesso rápido à documentação completa via tecla 'D'
- Melhorado diálogo de ajuda com informações de formato
- Aprimorado tratamento de erros e feedback do usuário

### Versões Anteriores

- Versão 0.0.7: Funcionalidade básica com zoom e modos de marcadores
- Versão 0.0.6: Implementação inicial com navegação de vídeo
