# Drible Arias (Palmeiras × Vasco): ângulos das mudanças de direção (2026-11-02)

Pedido da TV: ângulo e velocidade das duas mudanças de direção do drible ("a primeira de 180°
e a curva antes de finalizar a jogada").

## O que foi medido (`Lance_decifrado/pal_vas01nov/drible/enviar/resultados_drible_tv_mudancas_20261102/`)

| Mudança | Método | Ângulo | Tempo | Velocidade |
| --- | --- | --- | --- | --- |
| Meia-volta com a bola (pares 5–21) | giro da linha do quadril, SAM 3D Body, cada câmera | 173° (principal) / 183° (aberta), card ≈178° | 0,67 s, ~266 °/s | sai a ~13 km/h |
| Curva antes do desfecho (20–34 → 53–65) | cordas da trajetória, filtros 5/7/9 | 67–68°, card ≈68° | 0,79 s, ~85 °/s | entra ~12, quase para (2–7), sai ~16 km/h |

- **Por que o giro do corpo na meia-volta:** quase parado, a direção da trajetória perde sentido (a direção acumulada dá ~310° de volta). As duas câmeras concordam no giro do quadril (diferença de 10°).
- **A curva foi marcada pelo usuário** no vídeo equalizado, quadros 54–84 com quebra no 70. Isso corresponde aos originais 41–65, quebra 53 (`source_position − 1`).
- **Visto no plano do campo, o ângulo depende do recorte.** Só a quebra dá 27° e as pontas da janela dão 44°. A curva inteira, da corrida rente à lateral até a corrida para dentro do campo, dá ~65–68°. O usuário escolheu a curva inteira.
- **A câmera aberta, por conta própria, reproduz** 174° (cordas da meia-volta), 67° (curva) e 27° (quebra).
- **A dobra de ~97° nos pares 66–73 é a queda** com o Defensor 2. Ali os tornozelos não indicam direção de corrida, então essa dobra não é usada.

## Código

- `freekiki_openplay_kinematics.py`: `turn_between`, `chord`, `body_yaw_deg`, `body_rotation`, `cod_zone`, `speed_zone`. As zonas vêm de Monteiro, Santiago et al. 2025, *Sci Rep*.
- `freekiki_openplay_manifest.py`: `turns` no manifesto (`validate_turns`, `body_yaw_series`, `analyze_turns`, flag `stable`).
- `freekiki_openplay_video.py`:
  - `build_scenes`: o capítulo 28–40 s vira uma cena por mudança.
  - Card com o ângulo e os trechos no mapa.
  - `turn_figure` → `figuras/turns.png`.
  - Seção "Mudanças de direção" no relatório.
  - Referências novas: Monteiro et al. 2025, Marques et al. 2025 (*KSSTA*, corte de 90°) e Agência FAPESP 2025.

## Regenerar

```bash
.venv/bin/python -m vaila.freekiki openplay-report \
  --manifest <pacote>/manifest.json --output <pacote> --kind dribble --lang pt --animation
.venv/bin/python <pacote>/validar_entrega.py
```

A pasta `cameras/` foi copiada do pacote anterior. Por isso não foi preciso `--prepare-originals --rerun-sam`.
Pela `validacao_entrega.json`:
- A análise só ganhou `turns`.
- As entradas não mudaram.
- O vídeo tem 1800 quadros a 30 fps e decodifica inteiro.
- O PDF tem a seção e as referências novas.
