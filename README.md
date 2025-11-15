# 📌 Image Caption Generator

This project implements **two different image captioning methods** and provides a **Gradio-based interactive UI inside a Jupyter Notebook**.

## 🔹 Methods Implemented
1. **Pretrained Model Method**  
   Uses an off-the-shelf pretrained image captioning model (BLIP / ViT-GPT2 / other transformer models).

2. **Custom Transformer Model (Trained on Flickr8k)**  
   A fully trained encoder–decoder Transformer model fine-tuned on the Flickr8k dataset.  
   Includes:  
   ✔ Training scripts  
   ✔ Evaluation results (`evaluation_results.csv`)  
   ✔ Final trained model folder (`flickr8k-finetuned-model-final-*`)

---

# 📂 Project Title and Description

**Image Caption Generator using Pretrained & Transformer-based Approaches**

This project compares two popular methods for automatic image caption generation:
- Using a ready-made pretrained model.
- Using a custom-built Transformer trained on Flickr8k.

A **Gradio UI inside the Jupyter Notebook** allows users to upload images and generate captions from both models side-by-side.

---

# 🚀 Setup & Run Instructions (Step-by-Step)

## 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/image-caption-generator.git
cd image-caption-generator

2️⃣ Create and Activate a Virtual Environment
Windows
python -m venv venv
venv\Scripts\activate

Mac / Linux
python3 -m venv venv
source venv/bin/activate

3️⃣ Install Dependencies
pip install -r Code/requirements.txt

4️⃣ Download Required Models / Datasets
✔ Pretrained Model

Automatically downloaded by the Hugging Face pipeline when running the notebook.

✔ Flickr8k Trained Model

Place your trained model folder here:

/flickr8k-finetuned-model-final-20251115T0623/


Contents include:

encoder.pth

decoder.pth

tokenizer.pkl

config.json

other weights or metadata

✔ Dataset (Only if retraining)

Flickr8k Dataset:
Images: https://www.kaggle.com/datasets/adityajn105/flickr8k

Captions: https://www.kaggle.com/datasets/adityajn105/flickr8k

Dataset Size:

Images: ~1 GB

Captions: ~20 KB

Preprocessing Applied:

Tokenization

Vocabulary building

Removing rare words

Resizing images

Train/validation split

5️⃣ Running the Project (Notebook + Gradio UI)

Open Jupyter Notebook:

jupyter notebook


Run:

Code/image_caption_ui.ipynb


A Gradio UI will launch inside the notebook:

Upload an image

Generate captions with both models

Compare results in one interface

📁 Repository Structure
📦 image-caption-generator
│
├── 📁 Code
│   ├── Model_train.ipynb
│   ├── Model1_Transformer_Pipeline.ipynb
│   ├── Model2_image_captioning_Main.ipynb
│   ├── 
│
├── 📁 flickr8k-finetuned-model-final-20251115T0623/
│   ├── encoder.pth
│   ├── decoder.pth
│   ├── tokenizer.pkl
│   ├── config.json
│   └── (other assets)
│
├── evaluation_results.csv
│
└── README.md

🖼️ Example Input & Output
Input:

(Example: photo of a dog running on grass)

Output Captions
Model	Generated Caption
Pretrained Model	"A dog is running across a grassy field."
Transformer (Custom)	"A brown dog runs playfully over green grass."
📊 Evaluation

evaluation_results.csv includes metrics:

BLEU-1, BLEU-2, BLEU-3, BLEU-4

ROUGE

METEOR

CIDEr

These metrics compare both captioning methods.

🧰 Technologies Used
Languages

Python

Libraries

PyTorch

Hugging Face Transformers

torchvision

numpy

pandas

nltk

matplotlib

Pillow

Gradio

tqdm

Tools

Jupyter Notebook

CUDA/GPU (for training)

Kaggle (dataset source)
