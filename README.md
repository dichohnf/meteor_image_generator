# 🛡️ Meteor Image Generator — Generative Steganography on Images via VQ-GAN

![GitHub last commit](https://img.shields.io/github/last-commit/dichohnf/meteor_image_generator)
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)

Implementation of **SwE** (*Steganography without Embedding*), a generative image steganography protocol that extends the [Meteor](https://arxiv.org/abs/2310.16202) algorithm from the textual domain to **high-resolution digital images** using the **VQ-GAN** architecture.

> This repository is a fork of [CompVis/taming-transformers](https://github.com/CompVis/taming-transformers) (CVPR 2021), adapted to implement a steganographic embedding mechanism that hides secret messages directly into the autoregressive token generation pipeline of a VQ-GAN, without requiring training of a new model or modifying the learned probability distribution.

---

## Abstract

Traditional encryption protects the **content** of communications but is vulnerable to **traffic analysis**: the distinctive statistical fingerprints of encrypted data make it easily detectable by mass surveillance systems and deep packet inspection (DPI) firewalls.

**Steganography** addresses this gap by hiding the *very presence* of a message. Unlike classical LSB-based methods (vulnerable to statistical attacks), this project adopts a **generative paradigm**: the secret message is embedded **during image generation itself**, rather than modifying a pre-existing cover image.

The protocol works as follows:

1. A **VQ-GAN** (pre-trained on the target domain) encodes images into a discrete codebook of "visual tokens".
2. An autoregressive **Transformer** models the joint distribution over these tokens.
3. The secret message (encrypted and converted to bits) is embedded via **arithmetic coding**: for each patch position, the message bits select a specific token within the predicted probability distribution so that the generated image remains **statistically indistinguishable** from a naturally sampled one.
4. The receiver, with access to the same pre-trained model and shared secret (key/seed), can decode the message by re-computing the probability distributions and extracting the hidden bits.

---

## 🧠 Key Features

- **Steganography without Embedding (SwE)**: the secret message is encoded *during* image generation, not embedded afterwards into existing pixels.
- **No model retraining required**: uses pre-trained VQ-GAN checkpoints (e.g., S-FLCKR, FFHQ, CelebA-HQ, ImageNet, COCO-Stuff).
- **Statistical indistinguishability**: generated stego-images follow the same distribution as the original VQ-GAN samples, verified via FID scores and statistical tests.
- **Arithmetic coding‑based embedding**: message bits steer token selection without biasing the underlying autoregressive distribution.
- **Seed‑based reproducibility**: deterministic encoding/decoding given the same random seed and model checkpoint.
- **Batch encoding/decoding**: encode and decode multiple images with configurable message length and context fraction.

---

## 📁 Repository Structure

```
meteor_image_generator/
├── scripts/
│   ├── prova.py                # Main orchestrator: encode → decode pipeline
│   ├── encode_methods.py       # SteganographyEncoder: message hiding logic
│   ├── decode_methods.py       # SteganographyDecoder: message extraction logic
│   ├── input.py                # CLI argument parser and Options dataclass
│   ├── utils.py                # Bit manipulation, context building, model loading
│   ├── logger.py               # Logging utilities
│   └── stats.py                # Encoding/Decoding statistics collection
├── re-implementations/
│   └── VQGAN.py                # Simplified VQGAN from scratch (educational/experimental)
├── configs/                    # Training configurations (from original taming-transformers)
├── taming/                     # Core library (from original taming-transformers)
│   └── modules/
│       ├── vqvae/quantize.py   # Vector quantization modules
│       ├── transformer/mingpt.py # Autoregressive transformer (minGPT)
│       └── diffusionmodules/  # Encoder/decoder blocks
├── data/                       # Example data and dataset references
├── main.py                     # Original training entry point
├── setup.py                    # Package installation
└── envirorment.yaml            # Conda environment specification
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
conda env create -f envirorment.yaml
conda activate taming
pip install -e .
```

### 2. Download a Pre-trained VQ-GAN Model

Download a pre-trained VQ-GAN checkpoint (e.g., S-FLCKR) from the [original taming-transformers repository](https://github.com/CompVis/taming-transformers#overview-of-pretrained-models):

```bash
# Example: S-FLCKR
wget https://heibox.uni-heidelberg.de/d/73487ab6e5314cb5adba/ -O logs/2020-11-09T13-31-51_sflckr.zip
unzip logs/2020-11-09T13-31-51_sflckr.zip -d logs/
```

### 3. Run the Steganography Pipeline

The main entry point is `scripts/prova.py`. It:
1. Generates images from the VQ-GAN conditioned on a reference image.
2. Embeds a secret message during token selection (arithmetic coding).
3. Decodes the message from the generated stego-images.

```bash
python scripts/prova.py \
    --message "Your secret message here" \
    --model_directory logs/2020-11-09T13-31-51_sflckr/ \
    --seed 42 \
    --to_gen_number 10
```

**Options:**

| Argument | Description | Default |
|---|---|---|
| `--message` | Secret message to encode | `"Hello World!"` |
| `--model_directory` | Path to the VQ-GAN checkpoint directory | *required* |
| `--seed` | Random seed for reproducibility | `42` |
| `--to_gen_number` | Number of images to generate | `5` |
| `--output_directory` | Output directory | `output/` |
| `--context_fraction` | Fraction of rows used as conditioning context | `0.5` |
| `--quiet` | Suppress logging | `False` |
| `--random_generation` | Generate images without steganography (baseline) | `False` |

### 4. Output Structure

```
output/
├── meteor/
│   ├── rand_000.png            # Stego-image with embedded message
│   ├── rand_000_encoded_bits.txt
│   ├── rand_000_encoded_indices.txt
│   ├── rand_000_decoded_bits.txt
│   ├── rand_000_decoded_indices.txt
│   ├── rand_000_decoded_text.txt
│   └── rand_000_stats.json     # Encoding/decoding statistics
│   └── ...
└── random/
    └── ...                      # Baseline random images (no steganography)
```

---

## 🔬 How It Works

### Encoding Pipeline (`script/encode_methods.py`)

1. **Setup**: Load a pre-trained VQ-GAN and a reference image from the dataset.
2. **Context extraction**: The reference image is encoded into codebook indices (`z` space) and conditioning tokens (`c` space). The first fraction of rows serves as fixed context.
3. **Autoregressive token selection**: For each subsequent patch position `(row, col)`:
   - The Transformer predicts logits over the codebook given previous patches.
   - Logits are converted to probabilities and normalized to the codebook size.
   - **Arithmetic coding**: The message bits (converted to an integer) select a token within the cumulative probability distribution such that the message value falls in the corresponding probability interval.
   - The selected token is written to the building tensor.
4. **Self‑consistency check**: The selected token is verified by re-encoding the generated image patch — if the decoded index matches the selected token, the encoding is accepted.
5. **Decoding**: The decoder reconstructs the image from the built index tensor.

### Decoding Pipeline (`scripts/decode_methods.py`)

1. Load the stego-image and re-encode it to obtain the codebook indices.
2. For each patch position (same order as encoding):
   - Re-compute the Transformer probability distribution.
   - Find where the actual token falls in the cumulative distribution.
   - Extract the common prefix bits between the interval bounds.
3. Concatenate all extracted bits and convert back to text.

---

## 🧪 Experimental Results

Detailed experimental results, including FID comparisons between stego-images and natural VQ-GAN samples, statistical indistinguishability tests, and bit‑per‑token capacity analysis, are documented in the accompanying thesis:

> **"Steganografia generativa su immagini: adattamento del protocollo Meteor tramite l'architettura VQ-GAN"**  
> Matteo Diciotti — Università di Firenze, A.A. 2025/2026

---

## 📚 References

- **Original Taming Transformers paper**: [Esser et al., CVPR 2021](https://arxiv.org/abs/2012.09841) — [GitHub](https://github.com/CompVis/taming-transformers)
- **Meteor protocol (textual steganography)**: [arXiv:2310.16202](https://arxiv.org/abs/2310.16202) — [GitHub](https://github.com/dsokol465/Meteor)
- **VQ-GAN**: [Taming Transformers for High-Resolution Image Synthesis](https://compvis.github.io/taming-transformers/)
- **Thesis LaTeX sources**: [dichohnf/tesi_triennale](https://github.com/dichohnf/tesi_triennale)

---

## 📄 License

This project is licensed under the MIT License — see `License.txt` for details. The original [taming-transformers](https://github.com/CompVis/taming-transformers) code is © Patrick Esser, Robin Rombach, and Björn Ommer, licensed under MIT.