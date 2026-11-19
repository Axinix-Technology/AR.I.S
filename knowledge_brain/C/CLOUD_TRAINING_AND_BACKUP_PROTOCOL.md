# Cloud Auto-Backup & Google Drive Protocol

## 1. Overview & Purpose
In cloud environments like Google Colab, the local virtual machine storage (`/content/`) is ephemeral and automatically destroyed when sessions disconnect or idle. To prevent loss of trained model checkpoints, AR.I.S incorporates a **Cloud Auto-Backup Substrate** that automatically mirrors checkpoints directly to Google Drive upon **every periodic save** (e.g. every 500 steps).

## 2. Automatic Drive Discovery & Mirroring Mechanics
When training is initiated via `python run.py train-lm`:
1. **Auto-Discovery**: The system automatically checks if Google Drive is mounted at `/content/drive/MyDrive`.
2. **Dedicated Mirror Directory**: Creates `/content/drive/MyDrive/ARIS_backup/`.
3. **Save Synchronization**: At every checkpoint trigger (`session_step % save_every == 0` or target step completion), `ArisTrainer.save_checkpoint()` saves locally to `models/aris_lm_checkpoint.pt` and immediately executes an atomic copy (`shutil.copy2`) to `/content/drive/MyDrive/ARIS_backup/aris_lm_checkpoint.pt`.
4. **Auto-Restoration on Reconnect**: If a fresh Colab session starts and `models/aris_lm_checkpoint.pt` is missing from the local disk, the trainer and evaluation engine automatically detect the Google Drive backup, restore it into `models/`, and resume cumulative training without starting from step 0.

## 3. Explicit CLI Options
```bash
python run.py train-lm --steps 5000 --batch-size 48 --save-every 500 --backup-dir /content/drive/MyDrive/ARIS_backup
```
- If `--backup-dir` is not explicitly provided, it defaults to `/content/drive/MyDrive/ARIS_backup` whenever `/content/drive/MyDrive` is mounted.
