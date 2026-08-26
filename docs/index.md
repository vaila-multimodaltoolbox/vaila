# *vailá*

**Versatile Anarcho Integrated Liberation Ánalysis** — open-source Python 3.12 toolbox for multimodal biomechanical and movement analysis (IMU, MoCap, markerless 2D/3D, EMG, force plates, GNSS/GPS, and more), with a Tkinter-based desktop GUI.

---

## Start here — full script help list

Every module/tool help page (HTML + Markdown) is listed in one place:

### [**Open *vailá* Help Index →**](../vaila/help/index.html)

> Same path on disk: `vaila/help/index.html` · Markdown: [`vaila/help/index.md`](../vaila/help/index.md)

Regenerate the catalog after adding help pages:

```bash
uv run python bin/generate_help_index.py
```

---

## English — project overview

### What you get

- **Frame A — File manager:** rename, import/export, copy/move, tree, find, SSH transfer.
- **Frame B — Multimodal analysis:** IMU, MoCap, markerless, EMG, force plates, GNSS, and related pipelines.
- **Frame C — Tools:** CSV/C3D workflows, DLT 2D/3D reconstruction, video and image utilities, plots and visualization.

Optional stacks (CUDA/GPU templates, extras): root [`README.md`](../README.md) and [`AGENTS.md`](../AGENTS.md).

### Quick run (after install)

```bash
uv run vaila.py
```

### Guides (optional deep dives)

- [FIFA Skeletal Tracking Light workflow](fifa_workflow.md)
- [vaila-ElasticKick (VEK)](vek.md)
- [DLT 3D reconstruction and mesh alignment](dlt_reconstruction_and_mesh_alignment.md)
- [GUI button documentation](vaila_buttons/README.md)
- [PDF transcription (Brainstorm)](../vaila/help/transcribe_pdfs.md)
- [Hardware & GPU guide](../vaila/help/gpu_guide.md)

### Contributing

Pull requests and issues are welcome on GitHub. See [`CONTRIBUTING.md`](../CONTRIBUTING.md).

### License

Licensed under the **GNU Affero General Public License v3.0** (AGPL-3.0). See the `LICENSE` file.

### How to cite

```bibtex
@misc{vaila2024,
  title={vailá - Versatile Anarcho Integrated Liberation Ánalysis in Multimodal Toolbox},
  author={Paulo R. P. Santiago and Abel G. Chinaglia and others},
  year={2024},
  url={https://github.com/vaila-multimodaltoolbox/vaila}
}
```

---

## Português — visão geral do projeto

### Índice de ajuda dos módulos

Documentação HTML/Markdown por ferramenta: [`vaila/help/index.html`](../vaila/help/index.html).

### O que é o *vailá*

Caixa de ferramentas multimodal em Python 3.12 para análise do movimento e biomecânica, com interface desktop em Tkinter, integrando IMU, MoCap, rastreamento markerless 2D/3D, EMG, plataformas de força, GNSS/GPS e outros fluxos de dados.

- **Quadro A — Arquivos:** renomear, importar/exportar, copiar/mover, árvore, busca, SSH.
- **Quadro B — Análise multimodal:** IMU, MoCap, markerless, EMG, força, GNSS, etc.
- **Quadro C — Ferramentas:** CSV/C3D, DLT 2D/3D, vídeo/imagem, visualização.

### Executar após instalação

```bash
uv run vaila.py
```

### Guias

- [Workflow FIFA](fifa_workflow.md)
- [VEK](vek.md)
- [DLT 3D e alinhamento de mesh](dlt_reconstruction_and_mesh_alignment.md)
- [Botões da GUI](vaila_buttons/README.md)

### Contribuição / Licença / Citação

Ver seções em inglês acima (`CONTRIBUTING.md`, AGPL-3.0, BibTeX).

---

© 2026 *vailá* — documentation entry: `docs/help.html`
