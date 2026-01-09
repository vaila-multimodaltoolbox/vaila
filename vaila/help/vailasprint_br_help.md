# vailá Análise de Sprint (20m) - Ajuda

Bem-vindo ao módulo **vailá Sprint Analysis**. Esta ferramenta fornece uma análise biomecânica e de desempenho detalhada de sprints lineares de 20 metros, utilizando dados coletados pelo vailá Tracker. Ela foi projetada para processar múltiplas corridas automaticamente, gerar relatórios visuais e compilar um banco de dados para análise da equipe.

## 🚀 Fluxo de Trabalho

1.  **Iniciar**: Abra o `vaila.py`, verifique suas configurações e clique no botão **Sprint** (canto inferior esquerdo da interface principal).
2.  **Selecionar Modo**:
    - **Time Sprint (20m)**: Escolha esta opção para sprints lineares padrão de 20m.
    - *COD 90 Degree (20m)*: (Em Breve) Suporte para testes de Mudança de Direção.
3.  **Selecionar Pasta de Dados**: Escolha o diretório contendo seus arquivos de rastreamento `.toml`.
    - **Dica Importante**: Para extração automática de frames do vídeo (0m, 5m, etc.), certifique-se de que os arquivos de vídeo estejam na **mesma pasta** que os arquivos `.toml`, ou no caminho especificado dentro do TOML.
4.  **Processamento**: O script percorrerá cada arquivo `.toml` encontrado, calculará a cinemática e gerará os relatórios.
5.  **Saída**: Ao concluir, a pasta `vaila_sprint_reports` será aberta automaticamente.

---

## 📂 Estrutura de Saída (`vaila_sprint_reports`)

Todos os resultados são organizados para facilitar tanto o feedback individual quanto a análise em grupo.

### 1. Painel Principal (`general_report.html`)
**Público-Alvo: Técnico Principal, Preparador Físico**
- **Propósito**: Identificação de talentos e monitoramento da equipe.
- **Uso Prático**: Identificação rápida do jogador mais rápido do elenco. Use os **Rankings** para selecionar jogadores para funções táticas específicas (ex: pontas vs. zagueiros).
- **Banco de Dados Global** (`vaila_sprint_database.csv`): Arquivo mestre. **Uso Prático**: Preparadores físicos podem importar isso no Excel/PowerBI para monitorar o progresso ao longo da temporada ou comparar categorias (ex: Sub-17 vs Profissional).

### 2. Relatórios Individuais do Atleta
**Público-Alvo: O Atleta, Analista de Desempenho**
Uma subpasta dedicada é criada para cada análise (ex: `Silva_analysis...`). Dentro dela, você encontrará arquivos específicos:

#### A. O Relatório Interativo (`*_report_sprint20m.html`)
**O que é?** Um arquivo único contendo a análise visual completa da corrida.
**O que tem dentro?**
- **Curva de Velocidade**: Mostra *onde* a velocidade máxima foi atingida. No futebol, a aceleração inicial é frequentemente mais importante que a final.
- **Comparação com Usain Bolt**: Ferramenta educacional para comparar o perfil com a elite.
- **Evidência em Vídeo**: Frames extraídos aos 0m, 5m, 10m, 15m e 20m.
    - **0m**: Verificação da postura baixa de saída.
    - **5m**: Ângulo de ataque (aprox 45°).
    - **20m**: Postura ereta e mecânica de velocidade máxima.

#### B. Os Arquivos de Dados (`*_data.xlsx` / `*_data.csv`)
**O que são?** Dados numéricos brutos de cada parcial calculada.
**Colunas incluídas:**
1.  **distance_cumulative**: Distância da marcação (ex: 5.0, 10.0, 15.0, 20.0 metros).
2.  **duration**: Tempo gasto para cobrir aquele segmento específico.
3.  **speed_ms** & **speed_kmh**: Velocidade média naquele segmento.
4.  **acceleration_ms2**: Aceleração média naquele segmento.
**Uso Prático**:
- Importe no **Excel** para calcular métricas personalizadas como "Índice de Fadiga" (queda de velocidade).
- Compare especificamente o **split de 0-10m**, crucial para esportes multidirecionais.

#### C. As Imagens (`*.png`)
- **Gráficos**: Imagens em alta resolução das curvas de velocidade e aceleração (úteis para enviar via WhatsApp/Instagram).
- **Frames**: As imagens individuais extraídas do vídeo (0m, 5m, etc.).

---

## 📈 Entendendo as Métricas

### Velocidade (Speed)
- **Unidade**: Reportada em **km/h** (padrão para comunicação) e **m/s** (padrão científico).
- **Interpretação**:
    - **Velocidade Máxima**: A maior velocidade momentânea alcançada. Em um sprint de 20m, isso geralmente ocorre perto do final.
    - **Referência**: A velocidade de pico de Usain Bolt foi ~44.72 km/h (12.42 m/s). Jogadores de futebol de elite frequentemente atingem 32-36 km/h.

### Aceleração
- **Unidade**: Metros por segundo ao quadrado (m/s²).
- **Interpretação**: Quão rapidamente o atleta ganha velocidade.
    - **Fase de Partida (0-5m)**: deve mostrar os maiores valores de aceleração (potência explosiva).
    - **Fase de Transição**: A aceleração diminui à medida que a velocidade aumenta.
    - **Aceleração Zero**: Significa que o atleta atingiu sua velocidade constante máxima.

---

## 🛠 Solução de Problemas (Troubleshooting)

- **"No video frames extracted" (Nenhum frame extraído)**:
  - O script procura pelo nome do arquivo de vídeo salvo no `.toml`.
  - **Correção**: Copie os arquivos de vídeo originais (ex: `run1.mp4`) para a mesma pasta onde seus arquivos `.toml` estão localizados antes de rodar a análise.
- **Logo faltando**:
  - O relatório procura por `vaila.png` em `docs/images/` ou localmente. Garanta que a estrutura do projeto esteja intacta.
