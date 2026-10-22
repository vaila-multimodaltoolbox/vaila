# Graphify: grafo de código do vailá

Mapa de **estrutura do código** (quem chama quem, módulos, comunidades), gerado pelo
[graphify](https://github.com/Graphify-Labs/graphify) (`graphifyy` 0.9.93, Apache-2.0/MIT).
Complementa o `ai-memory` (memória das sessões): o grafo responde "o que depende de X?" sem ler arquivos grandes.

## Instalação (uma vez, fora do projeto)

```bash
uv tool install graphifyy==0.9.93     # dois "y"; não altera pyproject.toml nem uv.lock
graphify install --platform claude    # copia a skill para ~/.claude/skills/graphify
```

`graphify install` não registra hook no Claude Code; isso fica para depois, se for útil.

## Gerar o grafo (offline, só código)

O grafo do repositório inteiro não é gerado a partir da raiz (incluiria `.venv`, dados e `.local`). Use a lista
dos arquivos versionados:

```bash
git ls-files 'vaila/*.py' 'tests/*.py' | sort -u > /tmp/codefiles.txt
rsync -a --files-from=/tmp/codefiles.txt . /tmp/graph_src/
cd /tmp/graph_src
env -u ANTHROPIC_API_KEY -u OPENAI_API_KEY -u GEMINI_API_KEY graphify update . --no-cluster
env -u ANTHROPIC_API_KEY -u OPENAI_API_KEY -u GEMINI_API_KEY graphify cluster-only . --no-label --no-viz
cp -r graphify-out /home/preto/data/vaila/        # saída fica fora do git (.gitignore)
```

Tempo medido: ~11 s para 322 arquivos. Saída: `graph.json` (~12 MB) e `GRAPH_REPORT.md`. Nenhum LLM é usado (código
é analisado localmente com tree-sitter).

## Consultar

```bash
graphify god-nodes --top 10                      # hubs de arquitetura
graphify explain "build_sync_plan"               # chamadas de entrada e saída
graphify affected "visible_source_rect" --depth 2  # o que é impactado por uma mudança
graphify query "onde são calculados os cortes sincronizados?"
```

Limite conhecido: caminhos entre um módulo chamado por subprocesso (`vaila.py` → `run_vaila_module`) e a função
alvo não aparecem; o grafo não vê a fronteira do processo. Use `explain` e `affected` nesses casos.

## Regerar

Depois de mudanças grandes no código, repita o bloco "Gerar o grafo". Não há regeneração automática.
