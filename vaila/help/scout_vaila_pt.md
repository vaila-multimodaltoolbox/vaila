# vailá Scout - Ferramenta de Observação Esportiva

## Visão Geral

O vailá Scout é uma aplicação GUI integrada para anotar eventos esportivos em um campo de futebol virtual e gerar análises rápidas (ex: mapas de calor). É projetado para se adequar ao estilo do projeto vailá e não requer imagens externas do campo - o campo é desenhado em escala usando dimensões padrão FIFA (105m x 68m).

## Funcionalidades

### Funcionalidades Principais
- **Anotação em tempo real**: Clique no campo para marcar ações dos jogadores
- **Controle de timer**: Timer integrado com funcionalidades de iniciar/pausar/reiniciar
- **Gerenciamento de equipes**: Suporte para equipes da casa e visitante com escalações
- **Rastreamento de ações**: Tipos de ação personalizáveis com símbolos visuais
- **Rastreamento de resultados**: Resultados de sucesso/falha/neutro para cada ação
- **Geração de mapa de calor**: Análise visual dos padrões de movimento dos jogadores
- **Visualização dos scouts**: Análise visual das ações dos jogadores

### Gerenciamento de Dados
- **Importação/exportação CSV**: Carregar e salvar dados de eventos em formato CSV
- **Configuração TOML**: Sistema de configuração flexível para equipes, ações e configurações do campo
- **Conversão Skout**: Converter dados do formato de exportação ASCII do Skout.exe

### Interface do Usuário
- **Design responsivo**: Adapta-se a diferentes tamanhos de tela
- **Atalhos de teclado**: Acesso rápido a funções comuns
- **Feedback visual**: Eventos codificados por cor e marcadores de jogadores
- **Gerenciamento de escalação**: Seleção fácil de jogadores e troca de equipes

## Instalação

### Requisitos
- Python 3.x
- tkinter (GUI)
- matplotlib
- seaborn
- pandas
- rich (para impressões no console)
- toml (para escrita de config) e tomllib/tomli (para leitura de config)

### Executando a Aplicação

#### Opção 1: Da GUI do vailá
Clique no botão "Scout" na interface principal do vailá.

#### Opção 2: Linha de Comando
```bash
python vaila.py
```

#### Opção 3: Módulo Direto
```bash
python -m vaila.scout_vaila
```

#### Opção 4: Script Direto
```bash
cd vaila
python scout_vaila.py
```

## Guia de Uso

### Começando

1. **Inicie a aplicação** usando qualquer um dos métodos acima
2. **Carregue ou crie uma configuração** (arquivo TOML) com suas equipes e ações
3. **Selecione sua equipe** (use a tecla 'T' para alternar entre casa/visitante)
4. **Escolha uma ação** do menu suspenso
5. **Defina o resultado** (sucesso/falha/neutro)
6. **Clique no campo** para marcar ações dos jogadores
7. **Use o timer** para acompanhar o tempo do jogo

### Fluxo de Trabalho Básico

1. **Configure as Equipes**: Configure equipes da casa e visitante com escalações
2. **Defina as Ações**: Crie tipos de ação personalizados (passe, chute, desarme, etc.)
3. **Anote os Eventos**: Clique no campo para marcar ações dos jogadores
4. **Acompanhe o Tempo**: Use o timer integrado ou entrada manual de tempo
5. **Salve os Dados**: Exporte suas anotações para formato CSV
6. **Analise**: Gere mapas de calor e outras visualizações

### Interação com o Campo

- **Clique Esquerdo**: Marcar ação com resultado "sucesso"
- **Clique Direito**: Marcar ação com resultado "falha"  
- **Clique do Meio**: Marcar ação com resultado "neutro"
- **Ctrl + Clique**: Remover eventos próximos à posição clicada
- **Shift + Clique**: Marcar ação com resultado "falha" (alternativa para touchpad)
- **Alt + Clique**: Marcar ação com resultado "neutro" (alternativa para touchpad)

## Configuração

### Arquivo de Configuração TOML

A aplicação usa arquivos de configuração TOML para armazenar:
- Informações das equipes (nomes, cores, jogadores)
- Definições de ações (nomes, códigos, símbolos, cores)
- Configurações do campo (dimensões, unidades)
- Preferências visuais (tamanhos de marcadores, cores)

### Configuração Padrão

Quando iniciada pela primeira vez, a aplicação cria uma configuração padrão com:
- Campo padrão FIFA (105m x 68m)
- Duas equipes (CASA/VISITANTE) com jogadores de exemplo
- Ações comuns de futebol (passe, chute, desarme, etc.)
- Configurações visuais padrão

### Criando Configurações Personalizadas

1. Use "Criar Modelo" para gerar uma configuração base
2. Edite o arquivo TOML manualmente ou use as ferramentas de configuração
3. Carregue sua configuração personalizada
4. Salve as alterações conforme necessário

## Formato dos Dados

### Formato de Exportação CSV

Os eventos são exportados em formato CSV com as seguintes colunas:
```
timestamp_s, team, player_name, player, action, action_code, result, pos_x_m, pos_y_m
```

**Descrições das Colunas:**
- `timestamp_s`: Tempo em segundos desde o início do jogo
- `team`: Nome da equipe (CASA/VISITANTE ou personalizado)
- `player_name`: Nome completo do jogador
- `player`: Número do jogador
- `action`: Nome da ação (passe, chute, etc.)
- `action_code`: Código numérico da ação
- `result`: sucesso/falha/neutro
- `pos_x_m`: Posição X em metros (0-105)
- `pos_y_m`: Posição Y em metros (0-68)

### Sistema de Coordenadas

- **Origem**: Canto inferior esquerdo (0,0)
- **Eixo X**: Esquerda para direita (0-105m)
- **Eixo Y**: Baixo para cima (0-68m)
- **Unidades**: Metros (padrão FIFA)

## Atalhos de Teclado

| Atalho | Ação |
|--------|------|
| `Ctrl+S` | Salvar CSV |
| `Ctrl+O` | Carregar CSV |
| `Ctrl+K` | Limpar eventos |
| `H` | Mostrar mapa de calor |
| `V` | Visualização Scout |
| `R` | Reiniciar timer |
| `Espaço` | Iniciar/Pausar relógio |
| `?` | Abrir ajuda |
| `T` | Alternar equipe atual (casa/visitante) |
| `Ctrl+L` | Carregar configuração |
| `Ctrl+Shift+S` | Salvar configuração |
| `Ctrl+T` | Renomear equipes |
| `Dígitos 0–9` | Inserir código de ação; Enter aplicar; Backspace editar; Esc limpar |
| `Mouse` | Esquerdo=sucesso, Direito=falha, Meio=neutro |
| `Ctrl+Clique Direito` | Remover eventos próximos à posição clicada |

## Funcionalidades Avançadas

### Geração de Mapa de Calor

1. Clique no botão "Mapa de Calor" ou pressione 'H'
2. Selecione filtros de equipe e/ou jogador
3. Visualize o gráfico de densidade das ações dos jogadores
4. Analise padrões de movimento e zonas quentes

### Visualização do Scout

1. Clique no botão "Scout visualization" ou pressione 'V'
2. Selecione filtros de ação e/ou sucesso/falha
3. Ver o gráfico de ações no campo

### Conversão de Dados Skout

Converter dados do formato de exportação ASCII do Skout.exe:
1. Vá para **Ferramentas** → **Converter Skout para vailá**
2. Selecione seu arquivo Skout .txt
3. Escolha o diretório de saída
4. Digite o nome da equipe
5. Obtenha tanto arquivos CSV quanto de configuração TOML

### Gerenciamento de Jogadores

- **Visualização da Escalação**: Veja todos os jogadores de ambas as equipes
- **Seleção Rápida**: Clique nos nomes dos jogadores para selecioná-los
- **Numeração Automática**: Opções de atribuição automática de jogadores
- **Mapeamento de Nomes**: Vincular números dos jogadores aos nomes completos

### Funcionalidades do Timer

- **Timer ao Vivo**: Relógio do jogo em tempo real
- **Entrada Manual**: Definir timestamps específicos
- **Pausar/Retomar**: Controlar o tempo durante intervalos
- **Reiniciar**: Começar de novo com tempo limpo

## Solução de Problemas

### Problemas Comuns

**A aplicação não inicia:**
- Verifique a versão do Python (3.x necessário)
- Verifique se todas as dependências estão instaladas
- Certifique-se de que o tkinter está disponível

**Configuração não carrega:**
- Verifique a sintaxe do arquivo TOML
- Verifique as permissões do arquivo
- Use "Criar Modelo" para gerar config válida

**Campo não exibe:**
- Verifique a instalação do matplotlib
- Verifique as configurações de exibição
- Tente redimensionar a janela

**Dados não salvam:**
- Verifique as permissões de escrita no diretório de destino
- Verifique a compatibilidade do formato CSV
- Certifique-se de espaço suficiente em disco

### Dicas de Performance

- Feche outros aplicativos ao trabalhar com grandes conjuntos de dados
- Use dimensões apropriadas do campo para sua análise
- Limite o número de eventos simultâneos para melhor performance
- Salve o trabalho frequentemente para evitar perda de dados

## Estrutura de Arquivos

```
vaila/
├── scout_vaila.py          # Aplicação principal
├── help/
│   ├── scout_vaila.md      # Esta documentação
│   ├── scout_vaila.html    # Versão HTML
│   ├── scout_vaila_pt.md   # Documentação em português
│   └── scout_vaila_pt.html # Versão HTML em português
└── models/
    └── vaila_scout_config.toml  # Configuração padrão
```

## Licença

Este programa é software livre: você pode redistribuí-lo e/ou modificá-lo sob os termos da Licença Pública Geral GNU conforme publicada pela Free Software Foundation, seja a versão 3 da Licença, ou (a seu critério) qualquer versão posterior.

## Suporte

Para problemas, perguntas ou contribuições:
- **Email**: paulosantiago@usp.br
- **GitHub**: https://github.com/vaila-multimodaltoolbox/vaila
- **Documentação**: Veja os arquivos de ajuda no diretório vaila/help

---

**Versão**: 0.1.5  
**Última Atualização**: 22 de Agosto de 2025  
**Autor**: Paulo Roberto Pereira Santiago e Rafael Luiz Martins Monteiro
