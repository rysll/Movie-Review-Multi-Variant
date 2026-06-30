# NLP Model Variants: BERT · GPT-2 · Text-GAN
### Comparative Study on IMDB Movie Review Sentiment Classification & Generation

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Transformers-yellow)](https://huggingface.co)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Project Overview

This project implements and compares **three structurally distinct NLP architectures** on the same IMDB movie review corpus, evaluating their operational mechanics, performance limits, and behavioral characteristics across classification, language modeling, and adversarial generation tasks.

| Variant | Model | Task | Key Metric |
|---------|-------|------|-----------|
| **BERT** | `bert-base-uncased` | Sentiment Classification | Weighted F1 |
| **GPT-2** | `distilgpt2` | Causal Language Modeling | Perplexity + BLEU |
| **Text-GAN** | Custom CNN-based | Adversarial Text Generation | Discriminator Acc + BLEU |

---

## Dataset

**IMDB Movie Reviews** — Maas et al. (2011)
- **Hugging Face:** https://huggingface.co/datasets/stanfordnlp/imdb
- **Labels:** Negative (0), Positive (1)
- **Samples used:** 3,000 (1,500 per class, balanced)
- **Partitioning:** 80% train / 10% validation / 10% test (stratified, random_state=42)

```
Total samples : 3,000
Train         : 2,400 samples
Validation    :   300 samples
Test          :   300 samples
```

---

## Repository Structure

```
nlp-three-variants/
│
├── NLP_Three_Variants_BERT_GPT_GAN.ipynb   ← Main Colab notebook (all 3 models)
├── README.md                               ← This file
│
├── data/
│   ├── train.csv                           ← Training split (2,400 samples)
│   ├── val.csv                             ← Validation split (300 samples)
│   └── test.csv                            ← Test split (300 samples)
│   └── performance_comparison.csv          ← Final metrics table
│
├── figures/
│   ├── label_distribution.png             ← EDA — class balance across splits
│   ├── bert_training_curves.png           ← BERT loss & accuracy per epoch
│   ├── bert_confusion_matrix.png          ← BERT test set confusion matrix
│   ├── gpt_training_curves.png            ← GPT-2 loss & perplexity per epoch
│   ├── gan_training_curves.png            ← GAN generator/discriminator dynamics
│   ├── comparison_dashboard.png           ← 9-panel comparative dashboard
│   └── tokenization_comparison.png        ← Token count comparison across models
```

---

## Quickstart

### Google Colab (Recommended — requires GPU)
1. Upload `NLP_Three_Variants_BERT_GPT_GAN.ipynb` to [colab.research.google.com](https://colab.research.google.com/drive/1L5lcxW0a_zu2OCtW_QpHkniv_a1e5xol?usp=sharing)
2. Go to **Runtime → Change runtime type → T4 GPU** (or L4 with Colab Pro)
3. Run all cells: **Runtime → Run all**
4. Total estimated time: ~20 minutes on T4, ~10 minutes on L4

### Local Environment
```bash
git clone https://github.com/<your-username>/nlp-three-variants.git
cd nlp-three-variants
pip install torch transformers datasets evaluate accelerate nltk rouge-score \
            scikit-learn matplotlib seaborn pandas numpy tqdm sacrebleu
jupyter notebook NLP_Three_Variants_BERT_GPT_GAN.ipynb
```

---

## Model Architectures

### Variant 1 — BERT (Bidirectional Encoder Representations)

```
Input Review → [CLS] WordPiece Tokens [SEP]
             → BERT-base (12 layers, 768 hidden dim, 110M params)
             → [CLS] pooled output → Dropout → Linear(768 → 2) → Softmax
```

| Setting | Value |
|---------|-------|
| Tokenizer | WordPiece (vocab 30,522) |
| Max sequence length | 128 tokens |
| Optimizer | AdamW, lr=2e-5, weight_decay=0.01 |
| Scheduler | Linear warmup (10% of steps) |
| Epochs | 4 |
| Batch size | 32 |
| Loss | Cross-Entropy |

---

### Variant 2 — GPT-2 (Autoregressive Decoder)

```
<|sentiment|> + Review Text → BPE Tokens
             → DistilGPT-2 (6 layers, 768 hidden dim, 82M params)
             → Next-token logits (left-to-right causal attention)
```

| Setting | Value |
|---------|-------|
| Tokenizer | BPE (vocab 50,257 + 3 custom control tokens) |
| Control tokens | `<\|positive\|>`, `<\|negative\|>`, `<\|neutral\|>` |
| Max sequence length | 128 tokens |
| Optimizer | AdamW, lr=5e-5 |
| Epochs | 4 |
| Batch size | 16 |
| Inference | Nucleus sampling (top-p=0.9, temperature=0.8) |

---

### Variant 3 — Text-GAN (CNN-based Adversarial Network)

```
Generator:     Noise(100-dim) → FC → ConvTranspose1D ×3 → Logits(B, L, V)
                                           ↓ Gumbel-Softmax (straight-through)
Discriminator: Token embeddings → Conv1D(k=3,5,7) → GlobalMaxPool → Linear → P(real)
```

| Setting | Value |
|---------|-------|
| Tokenizer | Character-level (vocab ~100 chars) |
| Sequence length | 64 characters |
| Generator optimizer | Adam, lr=1e-3, β=(0.5, 0.999) |
| Discriminator optimizer | Adam, lr=1e-4, β=(0.5, 0.999) |
| Epochs | 30 |
| Batch size | 64 |
| N_critic | 2 (D trained 2× per G step) |

---

## Performance Results

### Actual Results (from executed notebook)

| Metric | BERT | GPT-2 | Text-GAN |
|--------|------|-------|----------|
| Precision (weighted) | **0.8410** | N/A | 0.9153 (discriminator) |
| Recall (weighted) | **0.8400** | N/A | 0.9367 (discriminator) |
| F1-Score (weighted) | **0.8399** | N/A | 0.9259 (discriminator) |
| BLEU Score | N/A | 0.0129 | 0.0000 |
| ROUGE-1 F1 | N/A | 0.2182 | 0.0099 |
| ROUGE-L F1 | N/A | 0.1155 | 0.0098 |
| Perplexity (test) | N/A | **51.06** | N/A |
| Avg Epoch Time | 55.2s | 58.5s | 0.8s |

> **Note on GAN discriminator F1:** A high discriminator F1 (0.93) means the generator is easily detectable — the ideal trained GAN should have discriminator accuracy near 0.5. This is a known limitation of character-level GANs trained on small datasets.

---

## Tokenization Comparison

Same sentence tokenized three different ways:

| Tokenizer | Strategy | Token Count | Example |
|-----------|----------|-------------|---------|
| BERT | WordPiece (subword) | ~11–25 | `['the', 'film', 'was', 'great', '##ly', 'over', '##rated']` |
| GPT-2 | Byte-Pair Encoding | ~9–22 | `['The', 'Ġfilm', 'Ġwas', 'Ġgreatly', 'Ġoverrated']` |
| Text-GAN | Character-level | ~40–80 | `['T','h','e',' ','f','i','l','m',' ','w','a','s'...]` |

Character tokenization produces sequences **4–5× longer** than subword methods, making it far harder for the GAN to model long-range dependencies.

---

## Key Findings

**BERT** achieved the highest task-specific performance (F1=0.84) because bidirectional attention + pretraining is ideal for understanding full-sentence semantics. However it cannot generate text.

**GPT-2** demonstrated strong domain adaptation after fine-tuning on IMDB reviews, with perplexity dropping from ~100 (pre-fine-tune) to 51 on the test set. Sentiment-conditioned generation worked qualitatively well.

**Text-GAN** showed the fundamental challenge of adversarial text generation: the Gumbel-Softmax approximation allows training but produces low-quality outputs (near-random character sequences), especially with a small 3,000-sample corpus and only 30 epochs.

---

## Analytical Discussion

### Why Text-GANs struggle with discrete text

Standard GANs require gradients to flow back through the generator's output. For images this is straightforward (continuous pixels). For text, selecting a discrete token (argmax) has zero gradient — making standard backpropagation impossible. We use **Gumbel-Softmax** as a differentiable relaxation, but it introduces a temperature hyperparameter that trades off between training stability and output quality.

GPT-2 sidesteps this entirely via **teacher forcing**: it receives ground-truth previous tokens during training and computes a clean cross-entropy loss at every step, giving stable low-variance gradients with no approximation needed.

### Why BLEU/ROUGE scores are low for GPT-2

These metrics measure n-gram overlap with a single reference sentence. For open-ended generation, there are thousands of valid continuations of a prompt — most sharing few n-grams with any one reference. A BLEU of 0.01 does not mean the model generates nonsense; it means the generated text is valid but lexically different from the specific reference. Human evaluation would score GPT-2 much higher.

---

## References

1. Maas, A., et al. (2011). *Learning word vectors for sentiment analysis.* ACL. — [Dataset](https://huggingface.co/datasets/stanfordnlp/imdb)
2. Devlin, J., et al. (2019). *BERT: Pre-training of deep bidirectional transformers.* NAACL. — [Model](https://huggingface.co/bert-base-uncased)
3. Radford, A., et al. (2019). *Language models are unsupervised multitask learners.* OpenAI. — [Model](https://huggingface.co/distilgpt2)
4. Yu, L., et al. (2017). *SeqGAN: Sequence generative adversarial nets with policy gradient.* AAAI.
5. Kim, Y. (2014). *Convolutional neural networks for sentence classification.* EMNLP.
6. Jang, E., Gu, S., & Poole, B. (2017). *Categorical reparameterization with Gumbel-Softmax.* ICLR.

---

## License

MIT License