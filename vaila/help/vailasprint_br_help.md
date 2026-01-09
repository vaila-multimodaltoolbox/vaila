# vaila Analise de Sprint (20m) - Ajuda

Bem-vindo ao modulo **vaila Sprint Analysis**. Esta ferramenta fornece uma analise biomecanica e de desempenho detalhada de sprints lineares de 20 metros e testes de Mudanca de Direcao (COD 180) utilizando dados coletados pelo vaila Tracker. Ela foi projetada para processar multiplas corridas automaticamente, gerar relatorios visuais e compilar um banco de dados para analise da equipe.

## Fluxo de Trabalho

1.  **Iniciar**: Abra o `vaila.py`, verifique suas configuracoes e clique no botao **Sprint** (canto inferior esquerdo da interface principal).
2.  **Selecionar Modo**:
    - **Time Sprint (20m)**: Escolha esta opcao para sprints lineares padrao de 20m.
    - **2X COD 180 degree (20m)**: Suporte para testes de Mudanca de Direcao (2x10m com giro de 180 graus).
3.  **Selecionar Pasta de Dados**: Escolha o diretorio contendo seus arquivos de rastreamento `.toml`.
    - **Dica Importante**: Para extracao automatica de frames do video (0m, 5m, etc.), certifique-se de que os arquivos de video estejam na **mesma pasta** que os arquivos `.toml`, ou no caminho especificado dentro do TOML.
4.  **Processamento**: O script percorrera cada arquivo `.toml` encontrado, calculara a cinematica e gerara os relatorios.
5.  **Saida**: Ao concluir, a pasta `vaila_sprint_reports` sera aberta automaticamente.

---

## Estrutura de Saida (`vaila_sprint_reports`)

Todos os resultados sao organizados para facilitar tanto o feedback individual quanto a analise em grupo.

### 1. Painel Principal (`general_report.html`)
**Publico-Alvo: Tecnico Principal, Preparador Fisico**

O relatorio geral agora inclui analise completa da equipe:

#### Banner de Estatisticas Principais
- **Contagem de Atletas**: Numero total de atletas analisados
- **Total de Corridas**: Numero de corridas processadas
- **Velocidade Maxima**: Velocidade maxima alcancada com **nome do atleta e ID da corrida**
- **Melhor Tempo**: Tempo mais rapido de 20m com **nome do atleta e ID da corrida**

#### Secao de Estatisticas da Equipe
- **Velocidade Media**: Velocidade media da equipe (km/h)
- **Desvio Padrao de Velocidade**: Variabilidade no desempenho da equipe
- **Tempo Medio**: Tempo medio da equipe (segundos)
- **Desvio Padrao de Tempo**: Medida de consistencia

#### Analise Visual de Desempenho
- **Grafico Dumbbell**: Compare Corrida 1 vs Corrida 2 para cada atleta. Linhas verdes = melhoria, Linhas vermelhas = queda.
- **Grafico de Dispersao de Melhoria**: Pontos acima da diagonal mostram melhoria da Corrida 1 para a Corrida 2.
- **Mapa de Calor de Desempenho**: Matriz completa de metricas com valores codificados por cores.

#### Analise de Clusters K-Means (3 Niveis)
Os atletas sao automaticamente classificados em 3 grupos de desempenho:
- **Alto Desempenho** (Verde): Atletas de elite
- **Desempenho Medio** (Laranja): Atletas na media
- **Baixo Desempenho** (Vermelho): Atletas que precisam de desenvolvimento

**Grafico Beeswarm**: Distribuicao visual dos atletas por cluster, mostrando pontos de dados individuais coloridos por nivel de desempenho.

#### Analise de Z-Score
Pontuacoes padronizadas mostrando como cada atleta se compara a media do grupo:
- **Verde Escuro (Z > 1.5)**: Excelente - significativamente acima da media
- **Verde (0.5 < Z < 1.5)**: Bom - acima da media
- **Amarelo (-0.5 < Z < 0.5)**: Media
- **Vermelho (-1.5 < Z < -0.5)**: Abaixo da media
- **Vermelho Escuro (Z < -1.5)**: Baixo - significativamente abaixo da media

**Z-Score Composto**: Pontuacao combinada para classificacao geral de desempenho.

#### Rankings de Desempenho
- **Ranking por Velocidade Maxima**: Atletas mais rapidos primeiro
- **Ranking por Tempo Total**: Tempos de conclusao mais rapidos primeiro

#### Banco de Dados Global (`vaila_sprint_database.csv`)
Arquivo mestre contendo todos os dados. **Uso Pratico**: Preparadores fisicos podem importar no Excel/PowerBI para monitorar o progresso ao longo da temporada ou comparar categorias (ex: Sub-17 vs Profissional).

### 2. Relatorios Individuais do Atleta
**Publico-Alvo: O Atleta, Analista de Desempenho**
Uma subpasta dedicada e criada para cada analise (ex: `Silva_analysis...`). Dentro dela, voce encontrara arquivos especificos:

#### A. O Relatorio Interativo (`*_report_sprint20m.html`)
**O que e?** Um arquivo unico contendo a analise visual completa da corrida.
**O que tem dentro?**
- **Curva de Velocidade**: Mostra *onde* a velocidade maxima foi atingida. No futebol, a aceleracao inicial e frequentemente mais importante que a final.
- **Comparacao com Usain Bolt**: Ferramenta educacional para comparar o perfil com a elite.
- **Evidencia em Video**: Frames extraidos aos 0m, 5m, 10m, 15m e 20m.
    - **0m**: Verificacao da postura baixa de saida.
    - **5m**: Angulo de ataque (aprox 45 graus).
    - **20m**: Postura ereta e mecanica de velocidade maxima.

#### B. Os Arquivos de Dados (`*_data.xlsx` / `*_data.csv`)
**O que sao?** Dados numericos brutos de cada parcial calculada.
**Colunas incluidas:**
1.  **distance_cumulative**: Distancia da marcacao (ex: 5.0, 10.0, 15.0, 20.0 metros).
2.  **duration**: Tempo gasto para cobrir aquele segmento especifico.
3.  **speed_ms** e **speed_kmh**: Velocidade media naquele segmento.
4.  **acceleration_ms2**: Aceleracao media naquele segmento.
**Uso Pratico**:
- Importe no **Excel** para calcular metricas personalizadas como "Indice de Fadiga" (queda de velocidade).
- Compare especificamente o **split de 0-10m**, crucial para esportes multidirecionais.

#### C. As Imagens (`*.png`)
- **Graficos**: Imagens em alta resolucao das curvas de velocidade e aceleracao (uteis para enviar via WhatsApp/Instagram).
- **Frames**: As imagens individuais extraidas do video (0m, 5m, etc.).

---

## Entendendo as Metricas

### Velocidade (Speed)
- **Unidade**: Reportada em **km/h** (padrao para comunicacao) e **m/s** (padrao cientifico).
- **Interpretacao**:
    - **Velocidade Maxima**: A maior velocidade momentanea alcancada. Em um sprint de 20m, isso geralmente ocorre perto do final.
    - **Referencia**: A velocidade de pico de Usain Bolt foi ~44.72 km/h (12.42 m/s). Jogadores de futebol de elite frequentemente atingem 32-36 km/h.

### Aceleracao
- **Unidade**: Metros por segundo ao quadrado (m/s2).
- **Interpretacao**: Quao rapidamente o atleta ganha velocidade.
    - **Fase de Partida (0-5m)**: deve mostrar os maiores valores de aceleracao (potencia explosiva).
    - **Fase de Transicao**: A aceleracao diminui a medida que a velocidade aumenta.
    - **Aceleracao Zero**: Significa que o atleta atingiu sua velocidade constante maxima.

### Z-Score
- **Unidade**: Desvios padrao da media.
- **Interpretacao**: Como um individuo se compara ao grupo.
    - **Z-Score Positivo**: Desempenho acima da media
    - **Z-Score Negativo**: Desempenho abaixo da media
    - **|Z| > 2**: Excepcional ou preocupante (outlier)

### Atribuicao de Cluster
- **Metodo**: Clusterizacao K-means em velocidade maxima e tempo total
- **Interpretacao**: Agrupamento baseado em dados de atletas similares
    - Util para criar grupos de treinamento
    - Identificar atletas prontos para promocao ou que precisam de suporte

---

## Solucao de Problemas (Troubleshooting)

- **"No video frames extracted" (Nenhum frame extraido)**:
  - O script procura pelo nome do arquivo de video salvo no `.toml`.
  - **Correcao**: Copie os arquivos de video originais (ex: `run1.mp4`) para a mesma pasta onde seus arquivos `.toml` estao localizados antes de rodar a analise.
- **Logo faltando**:
  - O relatorio procura por `vaila.png` em `docs/images/` ou localmente. Garanta que a estrutura do projeto esteja intacta.
- **Clusterizacao K-means desabilitada**:
  - Requer o pacote scikit-learn. Instale com: `pip install scikit-learn`
- **Perfil estatistico desabilitado**:
  - Requer o pacote ydata-profiling. Instale com: `pip install ydata-profiling`

---

## Dependencias

**Obrigatorias:**
- pandas, numpy, matplotlib, seaborn
- scipy (para calculos de Z-score)
- toml (para leitura de arquivos TOML)
- tkinter (para dialogos GUI)

**Opcionais:**
- opencv-python (cv2): Para extracao de frames de video
- scikit-learn: Para analise de clusterizacao K-means
- ydata-profiling: Para relatorios de perfil estatistico
