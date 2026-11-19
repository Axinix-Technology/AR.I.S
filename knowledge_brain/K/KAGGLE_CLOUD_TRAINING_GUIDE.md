# Kaggle Cloud GPU Training Guide (30 Hours/Week Free)

## 1. Overview
Kaggle provides **30 hours per week of free Dual Nvidia T4 or P100 GPUs**, 30 GB of RAM, and 73 GB of disk space. Because AR.I.S automatically detects CUDA (`torch.device("cuda")`), training on Kaggle runs **30x to 50x faster** than on a consumer laptop CPU, with zero heat or fan noise on your local machine.

---

## 2. Kaggle Notebook Setup

1. Go to [kaggle.com/code](https://www.kaggle.com/code) and click **"New Notebook"**.
2. In the right-hand **Notebook Options** panel:
   - **Accelerator**: Select **GPU T4 x 2** (or GPU P100).
   - **Internet**: Toggle to **ON** (Required to clone your repository and pull datasets).

---

## 3. Kaggle Execution Cells (Step by Step)

### Cell 1: Clone Repository & Install Dependencies
```bash
!git clone https://github.com/<your-username>/AR.I.S.git
%cd AR.I.S
!pip install -r requirements.txt
```

### Cell 2: Download Foundation Datasets (15 Seconds)
```bash
# Sourced from Hugging Face directly onto Kaggle's high-speed cloud connection
!python scratch/download_expanded_datasets.py
```

### Cell 3: (Optional) Bring Your Checkpoint or Custom Chat
If you want to continue from your laptop's current checkpoint:
1. On the right sidebar, click **"Add Input"** $\to$ **"Upload"** $\to$ select your `aris_lm_checkpoint.pt`.
2. Copy it into the project folder:
```bash
!cp /kaggle/input/<dataset-name>/aris_lm_checkpoint.pt models/
```

### Cell 4: Launch Blazing-Fast GPU Training
On a T4 GPU, batch sizes can be increased to 16 with zero accumulation delay:
```bash
# Balanced training across all pillars:
!python run.py train-lm --steps 5000 --batch-size 16 --accum-steps 2 --log-every 50 --save-every 500

# Or focus on Stage 3 & 4 Causal Reasoning:
!python run.py train-lm --focus reasoning --steps 5000 --batch-size 16 --accum-steps 2
```

### Cell 5: Evaluate the Milestone Scorecard
```bash
!python run.py eval --samples 5
```

### Cell 6: Prepare the Trained Model for 1-Click Download
Kaggle allows you to download any file placed in `/kaggle/working/`:
```bash
!cp models/aris_lm_checkpoint.pt /kaggle/working/
```

---

## 4. How to Download the `.pt` File from Kaggle to Your Laptop

1. In the right-hand panel under **Output** (`/kaggle/working/`), find `aris_lm_checkpoint.pt`.
2. Click the three dots (`...`) next to the file name.
3. Click **"Download"**.
4. Move the downloaded file into your laptop's `AR.I.S/models/aris_lm_checkpoint.pt`.

Now you can chat with your newly trained model 100% locally and offline on your laptop!
