---
name: new-skill
description: Cria novas skills do zero. Use quando o usuário quer criar uma skill, fazer uma skill, definir uma nova automação, ou transformar um workflow em skill. Também use para modificar ou melhorar skills existentes.
---

# New Skill

Workflow simplificado para criar novas skills do zero. Se o usuário quiser evals completos e benchmarks, use a skill `skill-creator` em vez desta.

## Passo 1: Capturar Intent

Se o usuário já mencionou o que quer na conversa atual, extraia da conversa. Se não, pergunte:

1. **O que a skill deve fazer?** (o que o Claude vai fazer com ela)
2. **Quando deve triggerar?** (frases/termos que o usuário vai usar)
3. **Qual o output esperado?** (arquivo, texto, ação, etc.)
4. **Precisa de casos de teste?** (opcional - só para skills com output verificável)

## Passo 2: Definir Nome e Descrição

Use este template para o frontmatter YAML:

```yaml
---
name: skill-name
description: O que faz + quando triggerar. Seja "pushy" - inclua contextos específicos onde deve ser usada, não apenas o nome da skill.
---
```

**Boa descrição:**
```
Extrai e analisa dados de planilhas Excel. Use quando o usuário menciona "excel", "spreadsheet", "xlsx", ou quer analisar dados tabulares, criar gráficos, ou calcular métricas de planilhas.
```

**Descrição fraca:**
```
Trabalha com Excel.
```

## Passo 3: Estruturar a SKILL.md

Crie a estrutura:

```
skill-name/
├── SKILL.md (obrigatório)
│   ├── YAML frontmatter (name, description)
│   └── Markdown com instruções
└── resources/ (opcional)
    ├── templates/
    ├── scripts/
    └── references/
```

### Corpo da SKILL.md

Organize assim:

```markdown
# Nome da Skill

Breve descrição do que faz (1-2 linhas).

## Quando Usar
Liste os contextos específicos onde deve triggerar.

## Como Usar
Passo a passo do workflow.

## Output
Formato esperado do resultado.

## Exemplos
Exemplos de prompts que devem funcionar.
```

## Passo 4: Escrever a Skill

**Diretrizes:**
- Use imperativo (faça X, não "você deveria fazer X")
- Explique o "porquê" das instruções importantes
- Mantenha sob 500 linhas
- Seja específica nos contextos de trigger
- Evite MUST/NEVER em caps - use explicações

## Passo 5: Revisão com Usuário

Mostre o draft e peça feedback:
- A descrição captura bem quando triggerar?
- As instruções cobrem os casos de uso principais?
- Algo faltando ou confuso?

## Estrutura de Diretório

Skills vivem em `.claude/skills/`. Para uma skill chamada `minha-skill`:

```
.claude/skills/minha-skill/
└── SKILL.md
```

## Exemplo Completo

Para criar uma skill que processa CSV:

1. **Name:** `csv-processor`
2. **Description:** Processa e limpa dados CSV. Use quando usuário menciona arquivos CSV, quer filtrar dados, remover linhas vazias, ou transformar dados tabulares.
3. **SKILL.md** com instruções de como ler, processar, e output.

Deseja criar uma skill agora? Me diga o que você quer automatizar.