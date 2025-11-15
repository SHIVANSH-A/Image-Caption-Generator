# Image Caption Generator

> **Image Caption Generator using Pretrained & Transformer-based Approaches**
>
> This single-file README contains everything you need to run the project: description, exact commands, folder layout, and usage examples. Copy–paste this file into your repo as `README.md` or open it inside your Jupyter workspace.

---

## 🔹 Project Overview

This project implements **two image captioning approaches** and exposes a **Gradio UI inside a Jupyter Notebook** so you can upload an image and compare captions side-by-side:

1. **Pretrained Model Method** — uses a Hugging Face / BLIP-style pretrained captioning model.
2. **Custom Transformer Method** — encoder–decoder Transformer fine-tuned on the Flickr8k dataset.

The repository also includes training scripts, evaluation results, and a final trained model folder for the Flickr8k model.

---

## 🗂️ Repository Structure

```
image-caption-generator/
├── Code/
│   ├── Model_train.ipynb
│   ├── Model1_Transformer_Pipeline.ipynb
│   ├── Model2_image_captioning_Main.ipynb  # Gradio UI notebook
├── flickr8k-finetuned-model-final-20251115T0623/
│   ├── encoder.pth
│   ├── decoder.pth
│   ├── tokenizer.pkl
│   ├── config.json
│   └── (other assets)
├── evaluation_results.csv
└── README.md      # this file
```

---

## ✅ What’s included

* Notebook-based code for training and inference (`Code/*.ipynb`).
* Evaluation metrics CSV comparing models (BLEU, ROUGE, METEOR, CIDEr).
* Final trained Flickr8k model in `flickr8k-finetuned-model-final-20251115T0623/`.

---

## 🔧 Setup — step-by-step commands

> Run these commands from your terminal inside the repository root `image-caption-generator/`.

### 1. Clone the repository

```bash
# Replace with your repository URL if needed
git clone https://github.com/your-username/image-caption-generator.git
cd image-caption-generator
```

### 2. Create & activate virtual environment

**Windows:**

```powershell
python -m venv venv
# activate
venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv venv
# activate
source venv/bin/activate
```

### 3. Install dependencies

```bash
# From repository root
pip install -r Code/requirements.txt
```

> If you don’t have `requirements.txt`, create one with the following minimal packages (example):

```text
# Code/requirements.txt (example)
torch>=1.13.0
torchvision
transformers
datasets
Pillow
numpy
pandas
nltk
matplotlib
gradio
tqdm
scikit-learn
joblib
```

### 4. Prepare models & datasets

**Pretrained model:**

* The notebook uses Hugging Face `transformers` / `from_pretrained()` which will download models automatically on first run (internet required).

**Custom Flickr8k model:**

* Place your trained model folder at the repository root with this exact path:

```
./flickr8k-finetuned-model-final-20251115T0623/
```

Folder must include:

```
encoder.pth
decoder.pth
tokenizer.pkl
config.json
(other optional metadata)
```

**Flickr8k dataset (only if retraining):**

* Images and caption text available from Kaggle (example link):

  * [https://www.kaggle.com/datasets/adityajn105/flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k)
* After downloading, place images in a folder and update the notebook paths used for training.

### 5. Optional: Prepare NLTK dependencies (tokenizers, punkt)

Run in Python once (inside your venv):

```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
```

---


## 🧾 Example Input & Output (format)

```
Input Image: dog_running.jpg

Output Captions:
Model                | Generated Caption
---------------------|-----------------------------------------
Pretrained Model     | "A dog is running across a grassy field."
Custom Transformer   | "A brown dog runs playfully over green grass."
```

---

## 📊 Evaluation

`evaluation_results.csv` contains per-model metrics such as:

* BLEU-1, BLEU-2, BLEU-3, BLEU-4
* ROUGE
* METEOR
* CIDEr

Example CSV header:

```csv
model,bleu1,bleu2,bleu3,bleu4,rouge,meteor,cider
pretrained,0.52,0.41,0.34,0.26,0.48,0.23,0.60
custom,0.47,0.39,0.31,0.23,0.45,0.21,0.55
```

---

## 🧠 Notes on the Custom Transformer

* Training pipeline included in `Code/Model_train.ipynb` and `Code/Model1_Transformer_Pipeline.ipynb`.
* Standard preprocessing used:

  * Tokenization (NLTK / custom tokenizer)
  * Vocabulary building with a frequency threshold
  * Image transforms (resize, normalization)
  * Train/validation split
* Save and load model weights using `torch.save()` and `torch.load()` for `encoder.pth` and `decoder.pth`.

---

## 💡 Tips & Troubleshooting

* **GPU training/inference:** If you have CUDA-enabled GPU, ensure `torch` detects it: `torch.cuda.is_available()`.
* **Model path errors:** Double-check the model folder path and filenames.
* **Hugging Face model download fails:** Make sure your environment has internet access and proper `transformers` version.
* **Gradio not launching inline:** If the notebook doesn’t show the Gradio widget inline, try `gradio.serve()` or run the notebook with `jupyter lab`.

---

