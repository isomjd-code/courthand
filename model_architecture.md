# Technical Report: Architecture Design for CP40 Latin Court Hand Recognition

## 1. Executive Summary

This project utilizes a customized **Convolutional Recurrent Neural Network (CRNN)** based on the PyLaia library to perform Handwritten Text Recognition (HTR) on 15th-century Latin Court Hand. The architecture deviates from standard HTR configurations (which typically prioritize speed on low-resource hardware) to leverage the high memory bandwidth and compute capability of an NVIDIA RTX 3090. The design prioritizes **spatial resolution** and **contextual depth** to address the specific paleographical challenges of Court Hand.

## 2. The Challenge: Latin Court Hand

The CP40 Court Hand presents three distinct challenges that standard HTR models often fail to capture:

1. **Vertical Density:** The script is characterized by "minims" (vertical strokes used for i, u, n, m) which are visually ambiguous without context.

2. **Vertical Extent:** The script features exaggerated ascenders (loops on 'l', 'b', 'h') and descenders (tails on 'g', 'p', 'q') that extend far beyond the x-height.

3. **Micro-Features:** Meaning often hinges on hairline suspension marks or subtle abbreviation symbols that are easily lost at lower resolutions.

## 3. Hardware Optimization (RTX 3090)

Standard HTR architectures are often constrained to run on consumer GPUs with 8GB VRAM. With the availability of an **RTX 3090 (24GB VRAM)**, we removed these constraints:

* **Input Resolution:** Increased input height by **100%** (from standard 64px to 128px).

* **Model Width:** Doubled the LSTM hidden units (from 256 to 512).

* **Throughput:** Utilized 97% compute saturation via large batch sizes and 16-bit mixed-precision training.

## 4. Architectural Specifications

### A. Input Processing

* **Fixed Height:** `128 pixels`

* **Rationale:** Standard 64px resizing causes aliasing that blurs thin suspension marks. 128px preserves the high-frequency spatial details required to distinguish a specific abbreviation from random noise or paper bleed-through.

### B. Convolutional Visual Encoder (CNN)

We implemented a **5-layer VGG-style** deep convolutional network.

* **Structure:** `[16, 32, 64, 128, 256]` feature maps.

* **Pooling Strategy:** `2x2` pooling applied 4 times.
  * *Vertical result:* Reduces 128px height to 8px.
  * *Final Collapse:* A global average pooling layer collapses the remaining 8px to 1px sequence.

* **Activation:** `LeakyReLU` was selected over ReLU to prevent "dead neurons" during the training of this deeper network.

* **Normalization:** Batch Normalization is applied at every layer to ensure stable gradient flow.

### C. Recurrent Sequence Modeler (RNN)

* **Type:** Bidirectional LSTM (Long Short-Term Memory).

* **Depth:** 3 Layers.

* **Hidden Units:** `512` (Standard is usually 256).

* **Rationale:** Court Hand is highly abbreviated. A visual shape often cannot be identified in isolation (e.g., distinguishing a minim stroke). The model must rely on the *grammatical context* of the Latin sentence to resolve visual ambiguity. Doubling the hidden units significantly increases the model's capacity to "memorize" Latin grammar and abbreviation patterns.

## 5. Training Configuration

To prevent the large model from overfitting the 10,000-line dataset, the following regularization strategies were employed:

* **Data Augmentation:** Random rotations (+/- 2°), elastic warping (simulating parchment distortion), and affine transformations were applied dynamically during training.

* **Weight Decay:** L2 penalty of `0.0001` to penalize overly complex weights.

* **Dropout:** 50% dropout applied to linear and recurrent layers.

* **Optimizer:** Adam (`lr=0.0003`) with ReduceLROnPlateau scheduling.

## 6. Conclusion

The resulting model contains approximately **23.6 million parameters**. This is significantly larger than typical historical HTR models (usually ~6-8 million parameters). This "High-Res / High-Capacity" approach is justified by the complexity of the Court Hand script and enabled by the specific availability of high-end GPU hardware.

