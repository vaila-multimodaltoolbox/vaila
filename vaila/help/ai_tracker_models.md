# ai_tracker_models

> English

## Module information

| Field | Value |
|-------|--------|
| **Category** | Tools |
| **File** | `vaila/ai_tracker_store.py`, `vaila/ai_tracker_hub.py`, `vaila/ai_tracker_train.py` |
| **Version** | 0.4.19 |
| **Updated** | 24 November 2026 |
| **Author** | Paulo Roberto Pereira Santiago |
| **GUI** | Yes (the **Net** button of `getpixelvideo.py`) |
| **CLI** | Yes (`python -m vaila.ai_tracker_hub ...`, `python -m vaila.ai_tracker_train --from-samples ...`) |

---

## Overview

This guide explains where the AI Track network of *vailá* is kept, how it is saved, and how to move it, retrain it, send it to Hugging Face and download it on another computer. The network is small, but it gets better with every marker you place by hand, so it should not be lost when you reinstall *vailá*.

The AI Track tool of `getpixelvideo.py` (button **AI Track**, key `T`) follows a marker from frame to frame. Each time you click, it learns. What it learns is the **network**: a set of small discriminators (`.npz`) and, if you ran a retrain, a fine-tuned backbone (`.pth`).

## What is saved, and where

The network lives in a folder you choose, outside the *vailá* install. Inside it, *vailá* uses the sub-folder `ai_tracker/`:

```text
<home>/ai_tracker/
  discriminators/   discriminator_<profile>__<key>.npz
  backbones/        <variant>_finetuned_<time>.pth
  samples/          <time>_<video name>.npz       (local only, never uploaded)
  history/          up to 10 previous versions of each file
  logs/             retrain logs
  manifest.json     active backbones, sha256 and size of each file, last retrain, Hugging Face sync
  README.txt
```

- **One discriminator per feature space.** The key `classic` is the 808 hand-made colour and region values (Deep off). A key like `resnet50-1a2b3c4d` is the same 808 values plus the deep features of one backbone (Deep on); `1a2b3c4d` is the start of the sha256 of that backbone file. A discriminator only works with that exact backbone. Before this change a session with Deep off overwrote the Deep checkpoint; now each space has its own file.
- **Samples.** AI Track keeps the points that you click while AI Track is on (a small patch around the point and a larger context crop), so a later retrain can learn from them. Only the points you click are recorded.
- **History.** Every time a file is replaced, the previous version goes to `history/<file name>/`. The last 10 are kept.
- **Official weights.** The ImageNet files (`*_imagenet.pth`) are not copied into the folder of the network. They are downloaded when Deep is first used.

## Choosing and moving the folder

*vailá* looks for the folder in this order:

1. `--ai-models DIR` (for example `uv run --no-sync vaila/getpixelvideo.py --ai-models ~/vaila_ai_models`);
2. the variable `VAILA_AI_MODELS_HOME`;
3. the file `~/.vaila/ai_models.json`, written when you choose a folder;
4. the old folder inside the install, `vaila/models/ai_tracker` (no copy outside it).

The first time AI Track saves something and no folder is chosen, `getpixelvideo` asks you for one once (suggestion `~/vaila_ai_models`). If you press Cancel, the old place inside *vailá* is used and the question comes back in the next session. Old checkpoints found in the old place are **copied** into the new folder (never moved) and put in the right feature space.

To choose or change the folder:

- in the Net window, **Change folder...**; or
- `uv run --no-sync python -m vaila.ai_tracker_hub set-home ~/vaila_ai_models`.

Choose a folder that you back up if you want a safe copy. To move an already chosen folder, copy it to the new place first, then choose the new place.

## When it is saved

The network is saved when you turn AI Track off, when you press Save (`Ctrl+S`), when the window closes, when you open another video, and just before the **Net** window opens, even with AI Track still on. A session where you placed no point by hand writes nothing and does not ask for a folder.

## The Net window

In `getpixelvideo`, press **Net** (right after **AI Track** and **Deep** in the toolbar). The same window opens from a terminal with `uv run --no-sync python -m vaila.ai_tracker_hub gui`. The text area shows the folder, the active backbones, each discriminator with its number of samples, the samples folder, the last retrain and the last Hugging Face sync. Every action also prints the equivalent terminal command (a line that starts with `>>`).

| Button | What it does |
|--------|--------------|
| **Change folder...** | Choose another folder for the network. Checkpoints found in the old place inside *vailá* are copied, never moved. A folder you chose before is not moved: copy it yourself. |
| **Open folder** | Opens the folder in the file manager. |
| **Download from Hugging Face** | Downloads the network of the repository written in the box above the buttons (no login). |
| **Use a local folder...** | Installs a downloaded copy or a folder on a USB stick. |
| **Retrain (GPU)** | Retrains from your samples in the background (see below). |
| **Send to Hugging Face...** | Uploads the network after you confirm the list of files (needs login). |
| **Restore previous version...** | Puts back the newest earlier version of a file. |
| **Refresh** | Reads the folder again. |

## Retrain

Retrain uses the samples you clicked to fine-tune the backbone and to rebuild the discriminators.

- Terminal: `uv run --no-sync python -m vaila.ai_tracker_hub retrain`, or directly `uv run --no-sync python -m vaila.ai_tracker_train --from-samples ~/vaila_ai_models/ai_tracker --mode both --device auto`.
- It trains only the last stage of the backbone plus a new two-class head (marker or background). `--unfreeze all` trains the whole network.
- It needs at least 40 training samples (`--min-samples`).
- **Hold-out without leakage.** With samples from two or more videos, whole videos are held out (about 20 %). With one video, the last block of frames is held out. The measure is the ROC AUC of a discriminator, on the held-out samples, with the old backbone and with the new one.
- **Activation rule.** The new backbone becomes the active one only if the AUC after is not lower than before. Otherwise the file stays in `backbones/` for inspection, the old backbone stays active, and the status is `kept_previous`.
- The discriminators are rebuilt with the same features the live tracker uses, so AI Track loads them.
- `--device auto` uses the GPU only if at least 3 GiB are free, else the CPU. Close SAM or other heavy jobs first.
- The result is in `manifest.json` (`last_retrain`) and the log is in `logs/retrain_<time>.log`.

See [ai_tracker_train](ai_tracker_train.md) for every option.

## Send the network to Hugging Face

1. Make a free account on huggingface.co and create an access token with **write** permission (Settings, Access Tokens).
2. Log in once on your computer:

   ```bash
   uv run --no-sync hf auth login
   ```

   Paste the token. You can also set the variable `HF_TOKEN` instead.
3. Look at what would be sent, without sending:

   ```bash
   uv run --no-sync python -m vaila.ai_tracker_hub push --dry-run
   ```

4. Send it (the same step is the **Send to Hugging Face...** button):

   ```bash
   uv run --no-sync python -m vaila.ai_tracker_hub push
   ```

   *vailá* lists the files and asks you to confirm. `--yes` skips the question.

What is sent: `manifest.json`, a model card `README.md`, `discriminators/*.npz` and the **active** fine-tuned backbones. What is never sent: `samples/`, `history/`, `logs/` (the samples are crops of people from your videos) and the official ImageNet weights.

The files are replaced in place in the repository. The Hugging Face history keeps every older version, so you can go back with `--revision`. The repository is public. The default is `paulosantiago/vaila-ai-tracker`; to use your own, add `--repo your-name/your-repo` or set `VAILA_AI_MODELS_REPO`.

## Download the network on another computer

No login is needed (the repository is public):

```bash
uv run --no-sync python -m vaila.ai_tracker_hub pull
uv run --no-sync python -m vaila.ai_tracker_hub pull --revision <commit>
```

Or press **Download from Hugging Face** in the Net window. *vailá* checks the sha256 of every file against the manifest first; one mismatch stops everything and nothing is installed. A file that is replaced goes to `history/` first. If a discriminator needs the official ImageNet weights that are not on the computer, *vailá* tells you; it downloads them the first time Deep is used.

To move the network without Hugging Face (a USB stick, a shared disk): copy the `<home>` folder, then on the new computer run `uv run --no-sync python -m vaila.ai_tracker_hub import <folder>` or use **Use a local folder...**.

## Go back to a previous version

Button **Restore previous version...** in the Net window, or:

```bash
uv run --no-sync python -m vaila.ai_tracker_hub restore ~/vaila_ai_models/ai_tracker/discriminators/discriminator_default__classic.npz
```

The newest earlier version in `history/` is put back, and the current one goes to `history/`, so a second call returns to where you were. To go back further, copy the file you want from `history/<file name>/`.

## Command reference

All commands accept `--ai-models DIR` before the command name to use another folder for that run.

| Task | Command |
|------|---------|
| See the folder, networks, samples and sync | `uv run --no-sync python -m vaila.ai_tracker_hub status` |
| Choose the folder | `uv run --no-sync python -m vaila.ai_tracker_hub set-home ~/vaila_ai_models` |
| Download | `uv run --no-sync python -m vaila.ai_tracker_hub pull [--repo R] [--revision X]` |
| Preview the upload | `uv run --no-sync python -m vaila.ai_tracker_hub push --dry-run` |
| Upload | `uv run --no-sync python -m vaila.ai_tracker_hub push [--repo R] [--yes]` |
| Install a local copy | `uv run --no-sync python -m vaila.ai_tracker_hub import <folder>` |
| Restore a file | `uv run --no-sync python -m vaila.ai_tracker_hub restore <file>` |
| Retrain in the background | `uv run --no-sync python -m vaila.ai_tracker_hub retrain` |
| Open the window | `uv run --no-sync python -m vaila.ai_tracker_hub gui` |
| Retrain in the foreground | `uv run --no-sync python -m vaila.ai_tracker_train --from-samples <home>/ai_tracker --mode both --device auto` |
| Retrain, only the plan | `uv run --no-sync python -m vaila.ai_tracker_train --from-samples <home>/ai_tracker --dry-run` |
| Login to Hugging Face | `uv run --no-sync hf auth login` |

## Privacy

- The samples hold image crops of people from your videos. They stay in your folder and are never uploaded. Do not copy `samples/` to a public place.
- A discriminator or a backbone file is a set of numbers. It is not a picture, but it was trained on your footage: send it only if you are allowed to share what the footage shows.
- The repository is public: anyone can download what you push.

## Limits

- The upload to Hugging Face was written and tested with a stand-in service. A real upload has not been exercised yet; it will be tried at the owner's first push, so treat that first push as a test (use `--dry-run` first).
- The gain of the retrain was measured only on two benchmark videos. Expect it to help on similar footage and check the tracked points on yours; the activation rule stops a worse network, but a held-out score is not a guarantee.
- A discriminator trained with Deep needs the same backbone file (same sha256) to be used.

## See also

- [getpixelvideo](getpixelvideo.md) (AI Track and the Net button)
- [ai_tracker_train](ai_tracker_train.md) (every training option)
