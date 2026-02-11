# interp_smooth_split

## 📋 Module Information

- **Category:** Processing
- **File:** `vaila/interp_smooth_split.py`
- **Version:** 0.1.0
- **Author:** Paulo R. P. Santiago
- **GUI:** ✅ | **CLI:** ✅

## 📖 Description

Ferramenta para preencher dados em falta em ficheiros CSV (interpolação), suavização e divisão de dados. Destinada a análise biomecânica e séries temporais.

### Funcionalidades principais

1. **Interpolação (preenchimento de gaps)**  
   Linear, cúbica, nearest, Kalman, Hampel; ou nenhuma / skip.

2. **Suavização**  
   Nenhuma, Savitzky-Golay, LOWESS, Kalman, Butterworth, Splines, ARIMA, mediana móvel.

3. **Configuração em TOML**  
   `smooth_config.toml`: fonte única de verdade. Ao aplicar no diálogo o ficheiro é gravado; a análise de qualidade e o processamento (GUI ou CLI) usam estes valores quando o ficheiro existir. O diretório de output também recebe uma cópia do TOML usado.

4. **Análise de qualidade**  
   Botão "Analyze Quality": CSV de teste, análise por coluna, Winter residual (fc 1–15 Hz, fs em Hz). Selecção de coluna com Combobox (lista permanece aberta até escolher).

5. **Padding** e **divisão de dados** configuráveis.

---

## Como executar

### GUI (por defeito)

- **Módulo:**  
  `python -m vaila.interp_smooth_split`

- **Script:**  
  `python vaila/interp_smooth_split.py`

- **Forçar GUI com argumentos:**  
  `python -m vaila.interp_smooth_split --gui`

Abre o diálogo de configuração; após Apply escolhe-se o diretório de origem. O output é escrito num subdir com timestamp (ex.: `processed_linear_lowess_YYYYMMDD_HHMMSS`).

### CLI (linha de comando)

A configuração é lida, por ordem de prioridade, de:

1. `--config` / `-c` (caminho para um TOML)
2. `smooth_config.toml` no diretório de entrada
3. `smooth_config.toml` no diretório atual

Se não for encontrado nenhum config, o script termina com erro (pode criar um TOML via Apply na GUI ou usar um template).

**Argumentos:**

| Argumento | Descrição |
|-----------|-----------|
| `-i`, `--input` | Diretório com ficheiros CSV (obrigatório em modo CLI) |
| `-o`, `--output` | Diretório de saída (opcional; por defeito é criado um subdir com timestamp dentro de `--input`) |
| `-c`, `--config` | Caminho para `smooth_config.toml` (opcional) |
| `--gui` | Abre a interface gráfica em vez de correr em CLI |

**Exemplos:**

```bash
# Usar smooth_config.toml no diretório de entrada ou no cwd
python -m vaila.interp_smooth_split --input ./data

# Indicar diretório de saída e ficheiro de config
python -m vaila.interp_smooth_split -i ./data -o ./results -c ./smooth_config.toml

# Só indicar entrada; output = subdir com timestamp dentro de ./data
python -m vaila.interp_smooth_split -i ./data
```

---

## Ficheiro de configuração (TOML)

- **smooth_config.toml** (gravado ao aplicar no diálogo ou na pasta de output de cada run):
  - `[interpolation]`: method, max_gap
  - `[smoothing]`: method e parâmetros (frac, it para LOWESS; cutoff, fs para Butterworth; etc.)
  - `[padding]`: percent
  - `[split]`: enabled
  - `[time_column]`: sample_rate

---

## 🔧 Funções principais

- `run_fill_split_dialog` — Abre o diálogo GUI e, após Apply, processamento em batch.
- `run_batch(source_dir, config, dest_dir=None, use_messagebox=True)` — Processa todos os CSV em `source_dir` com a config dada; usado pela GUI e pela CLI.
- `process_file` — Processa um CSV com a configuração fornecida.
- `load_smooth_config_for_analysis` / `save_smooth_config_toml` — Leitura/gravação de `smooth_config.toml`.
- `winter_residual_analysis` — Análise de resíduos Winter (Butterworth, RMS, sugestão de fc).

---

📅 **Atualizado:** 2026  
🔗 **Parte do vailá - Multimodal Toolbox**  
🌐 [Repositório GitHub](https://github.com/vaila-multimodaltoolbox/vaila)
