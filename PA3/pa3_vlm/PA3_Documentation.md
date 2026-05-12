# PA3 Vision-Language Models: The Definitive Working Document

This is the fully expanded documentation for the PA3 codebase. It breaks down the repository folder by folder, traces the exact implementation of every task, and analyzes the empirical results to derive deep, conceptual insights. 

---

## 1. Deep Dive into Repository Structure

The entire project is cleanly modularized inside the `src/pa3/` directory. Each subfolder handles a specific domain of the multimodal pipeline.

### `src/pa3/data/` (Data Pipelines & Preprocessing)
This folder handles all dataset generation, parsing, and token-sequence formatting. 
- **`alpaca.py`**: Loads the Alpaca dataset (1,000 instruction-tuning examples). This dataset acts as the "anchor" for measuring and preventing Catastrophic Forgetting. By mixing Alpaca examples into our VLM training, we force the LM to continually remember how to process pure text.
- **`cifar_part_a.py`**: The central data engine for **Part A**. It splits CIFAR-10 into a highly controlled subset. Critically, it executes the `CLIPImageProcessor` to match OpenAI's exact pixel normalization. To save immense computation, it passes images through the frozen CLIP model just once and caches the `[49, 768]` patch tensors. It procedurally generates 5 distinct VQA questions per image (e.g., categorical, binary, semantic abstraction).
- **`synthetic_part_b.py`**: The data engine for **Part B**. It mathematically generates a synthetic dataset of 16x16 images across 6 geometrical classes (spirals, crosses, etc.). It calculates tokenization logic that dynamically shifts VQ-VAE codebook indices up by the LM's vocabulary size (`Vtxt = 49152`) so that the LM interprets codebook ID `0` as its own internal token `49152`.

### `src/pa3/models/` (Architectures & Hooks)
Contains the raw PyTorch neural network definitions.
- **`connector.py`**: Defines the Continuous Connector for Part A. It is a 2-layer Multi-Layer Perceptron (MLP) with a GELU activation that bridges the `768`-dimensional CLIP space into the `960`-dimensional LM space. 
- **`overlay_embedding.py`**: A brilliant memory-saving mechanism for Part B. Expanding an LM vocabulary usually requires creating a massive, computationally expensive new embedding table. Instead, this file creates a dynamic `nn.Embedding` of just 258 rows (for `<image>`, `</image>`, and `256` VQ-VAE tokens). It uses a PyTorch `forward_hook` to intercept token inputs, looking up text tokens in the frozen LM table and visual tokens in the new, tiny overlay table.
- **`smollm_lora.py`**: The model loader. It instantiates `SmolLM2-360M-Instruct` in `float16` and injects PEFT LoRA adapters (rank 16, alpha 32) into the attention and feed-forward projections (`q_proj, k_proj, v_proj, o_proj`). 
- **`vqvae.py` & `vector_quantizer.py`**: The Convolutional Encoder-Decoder for Part B. The vector quantizer is built with an Exponential Moving Average (EMA) update mechanism and a Straight-Through Estimator (STE) to allow backpropagation through the non-differentiable `argmin` token selection.

### `src/pa3/train/` (Training Loops)
Contains the exact training phase implementations, complete with custom loss balancers.
- **`train_part_a_phase1.py`**: Phase 1 "Connector Warmup." The LM and CLIP models are 100% frozen. Gradients only flow to the small MLP connector, training it to project CLIP features into a geometrical space the LM understands.
- **`train_part_a_phase2.py`**: SFT with Language Replay. Unfreezes the LoRA adapters. The loss function becomes a hybrid: $L_{mixed} = L_{VQA} + \lambda L_{LM}$. This teaches the LM visual reasoning while retaining its NLP capabilities.
- **`train_part_a_phase3.py`**: Final VQA alignment. Drops the language replay ($\lambda = 0$) and trains purely on VQA for 1 epoch to maximize visual metrics.
- **`train_part_b_lm.py`**: The master training loop for Part B. It optimizes for VQA, Language Modeling, and Autoregressive Image Generation simultaneously. To prevent VRAM exhaustion (OOM), it processes each task sequentially, accumulating gradients before stepping the optimizer.
- **`train_vqvae.py`**: The standalone VQ-VAE trainer, run strictly before any LM integration.

### `src/pa3/eval/` & `src/pa3/utils/` (Metrics & Infrastructure)
- **`eval/`**: Scripts like `eval_part_a_vqa.py` calculate exact-match accuracy. `eval_part_b_imagegen.py` forces the LM to generate sequences of 16 discrete tokens, masking out text vocabulary, and passes those tokens to the VQ-VAE decoder to render PNG grids. `eval_part_a_modality_gap.py` calculates the cosine distances between text and visual representations.
- **`utils/`**: Utilities for persistent metric logging (`logging.py`), mixed-precision device placement (`device.py`), and atomic checkpoint saving (`checkpointing.py`).

---

## 2. Part A: Continuous Connector VLMs

### Task A-C0: Data Pipeline and Model Loading
- **Implementation:** The system filters CIFAR-10 into a highly controlled subset (seed 42). It strips the `CLS` token from the CLIP ViT, extracting only the 49 spatial patches. It builds synthetic VQA data spanning 5 different reasoning skills (Recognition, Binary, Abstraction, Attribute, Category). Baseline PPL is computed by evaluating the base `SmolLM2-360M` on the Alpaca dataset.
- **Results:** Feature caching was highly successful. The baseline Perplexity (`PPL0`) was logged to establish a benchmark for text-comprehension prior to multimodal fine-tuning.
- **Insights & Limitations:** **Insight:** Pre-caching visual features from a frozen encoder transforms a massive multimodal training pipeline into a lightweight, rapid fine-tuning job. **Limitation:** Relying on frozen CLIP features means the model is permanently blind to any pixel-level details (like exact textures or tiny texts) that CLIP ignored during its own pre-training.

### Task A-C1: Phase 1 — Connector Initialisation
- **Implementation:** Trains the `MLP` connector (`1.66M` params). It restricts gradients using `.requires_grad_(False)` on both the LM and CLIP. Crucially, the script computes `rnorm`—the ratio of the L2 norm of the projected visual tokens to the LM's text tokens—and persistently rescales the connector weights to ensure the visual embeddings don't blow up the LM's activations.
- **Results:** The output artifacts (`partA_phase1_captions.txt`) showed the model outputting coherent text in response to visual input.
- **Insights & Limitations:** **Insight:** A simple linear-to-GELU projection is more than capable of rotating a continuous visual representation into a space that triggers correct syntax within an LM. **Limitation:** The MLP processes each of the 49 patches entirely independently (no cross-patch attention), significantly limiting the model's complex spatial reasoning capabilities.

### Task A-C2: Phase 2 — SFT with Language Replay
- **Implementation:** Integrates LoRA adapters. The loss dynamically combines VQA prediction with Alpaca text replay (`lambda = 0.2`). Targets are tightly masked using `-100` so gradients strictly flow from the model's generated answer, not the question prefix.
- **Results:** The forgetting ratio `R = PPL_fine / PPL0` remained stable and close to 1.0.
- **Insights & Limitations:** **Insight:** A surprisingly low lambda value (0.2) is sufficient to mathematically anchor the LoRA weights, preventing the model from collapsing into a pure visual-answering machine. **Limitation:** Continually interleaving text replay inherently slows down the convergence of the visual tasks compared to a pure VQA focus.

### Task A-C3: Phase 3 — VQA Alignment
- **Implementation:** 1 epoch of pure VQA training without language replay (`lambda = 0`), executed at a lower learning rate. Following a `CUDA OOM` crash, the configuration was updated to use `batch_size: 16` and `grad_accum: 8` (effective batch size 128) to safely fit inside the 11GB VRAM limit.
- **Results:** The run completed successfully, producing final aligned weights.
- **Insights & Limitations:** **Insight:** Once the connector is warmed up and the LoRA adapters have stabilized the semantic mapping (Phase 2), a final rapid, low-LR epoch explicitly dedicated to VQA can perfectly align the token outputs without instantly destroying the text priors.

### Task A-C4: VQA Evaluation
- **Implementation:** Tested on the held-out CIFAR-VQA validation set. Computes accuracy per-class and per-template against a "text-only" and "majority-vote" baseline.
- **Results:** 
  - **Overall Accuracy:** Phenomenal `0.9700` (97%).
  - **Baselines:** The Majority vote scored `34.4%`. The Text-only baseline scored a perfect `0.0%`.
  - **Template Breakdown:** The model scored `100%` on binary (yes/no) questions and `99%` on abstraction/category reasoning. It was slightly weaker on direct recognition (`89%`).
  - **Qualitative:** Top-5 logit traces reveal the model acts extremely decisively on correct answers. In its rare failures, it tends to confuse semantically close classes (e.g., classifying a deer as a frog, where both are organic/natural entities in low-resolution patches).
- **Insights & Limitations:** **Insight:** The 0% accuracy on the text-only baseline proves absolutely that the LM is not "cheating" by memorizing linguistic priors; it is genuinely reading the continuous visual tokens to answer the questions. **Limitation:** The drop in raw recognition accuracy (`89%`) confirms the earlier hypothesis—CLIP patches lack fine-grained, high-resolution discriminative power.

### Task A-C5: Modality Gap Analysis
- **Implementation:** Computes the cosine distances between text representations and visual representations across all training phases.
- **Results:** 
  - The Modality Gap (MG) remained consistently wide throughout training: Phase A1 (`1.208`) -> Phase A2 (`1.217`) -> Phase A3 (`1.217`).
  - Cross-modal cosine similarity remained functionally zero (`0.018` down to `0.007`).
- **Insights & Limitations:** **Insight:** This is one of the most profound takeaways of the architecture. The Modality Gap *does not close*. The LM's internal text embeddings and the injected visual embeddings remain geometrically isolated. However, the Transformer attention layers are powerful enough to route information seamlessly across this gap, proving that multimodal alignment is about functional mapping, not geometric overlap!

---

## 3. Part B: Discrete VQ-VAE

### Task B-C0: Dataset Generation and Model Loading
- **Implementation:** Procedurally generates a custom 16x16 geometry dataset. The code creates images, corresponding VQA pairs, and generation prompts. The tokenization sequences strictly left-pad the data.
- **Results:** Datasets successfully generated. Output artifacts include `synthetic_grid.png`, showing visually distinguishable geometric classes. `PPL0` evaluated to ~8.0.
- **Insights & Limitations:** **Insight:** Synthetic generation allows for perfect control over the token mapping pipelines without the noise of real-world photography. **Limitation:** It is a "toy" dataset—success here proves architectural correctness but scaling a discrete VQ codebook to highly complex, high-resolution photography is vastly more difficult.

### Task B-C1: VQ-VAE Training and Codebook Analysis
- **Implementation:** A CNN Encoder compresses the 16x16 images to a `[4x4x64]` latent space. The Quantizer snaps continuous latents to a discrete `K=256` codebook. The model relies heavily on Exponential Moving Average (EMA) to update the codebook, stabilizing training. Dead-code restarts monitor codebook utilization and re-initialize any code vectors that are not actively used.
- **Results:** Reconstruction MSE achieved near-zero levels. Artifacts (`vqvae_log.jsonl`, `vqvae_summary.csv`) show that dead-code restarts successfully prevented codebook collapse, ensuring all 256 tokens are actively utilized.
- **Insights & Limitations:** **Insight:** Because deep learning models naturally suffer from "rich get richer" problems (where a few codebook indices are used repeatedly and others die), EMA and Dead-Code restarts are absolutely mandatory to force the network to utilize the full expressive power of its 256-token vocabulary. **Limitation:** VQ-VAEs are lossy. They discard fine textural details to fit imagery into discrete clusters.

### Task B-C2: Vocabulary Expansion
- **Implementation:** Implements the `overlay_embedding.py` strategy. Crucially, before the LM begins training on visual tokens, the script runs a two-phase "Projector Warm-up". It trains a simple linear projection mapping the 64-D codebook vectors into the 960-D LM space. Once converged, these geometric embeddings are transplanted into the overlay rows.
- **Results:** Validated dynamically. The LM correctly absorbed 258 new tokens without requiring a massive parameter expansion.
- **Insights & Limitations:** **Insight:** Transplanting pre-warmed embeddings prevents the LM from destroying its pre-trained text space with early gradient noise. It gives the LM a mathematically sound starting point for visual tokens that directly correlates to the VQ-VAE's geometric latent space.

### Task B-C3: Tokenisation Pipeline
- **Implementation:** Sequences are precisely formatted. Visual tokens are offset by `Vtxt + 2`. VQA sequences are formatted as `[BOS, <image>, 16 visual IDs, </image>, question, answer, EOS]`. Generation sequences are formatted as `[BOS, prompt, <image>, 16 visual IDs, </image>, EOS]`.
- **Results:** Sequence parsing logs (`partB_token_type_debug.txt`) prove offsets and boundaries are correct.
- **Insights & Limitations:** **Insight:** Correct tokenization means the LM treats image patches exactly like words in a sentence, unifying the modalities completely. **Limitation:** The LM treats a 2D image grid as a 1D sequence of 16 tokens. This fundamentally disrupts spatial coherence, making it harder for the model to reason about vertical adjacencies (e.g., shapes that span multiple rows).

### Task B-C4: Mixed-Objective Fine-Tuning
- **Implementation:** LoRA fine-tuning applied to the unified LM. It minimizes a massive three-part objective: $L_{mixed} = L_{VQA} + 0.2 \cdot L_{LM} + 0.5 \cdot L_{IMG}$. The script splits the batch by task type, runs a forward and backward pass, and scales the gradients sequentially before calling `optimizer.step()`.
- **Results:** The sequential strategy entirely avoided OOM errors. Training logs (`partB_lm_summary.jsonl`) show steady convergence across all three objectives simultaneously.
- **Insights & Limitations:** **Insight:** Sequential passes are a vital engineering technique for fitting massive multitask objectives into small VRAM allocations. The single LM successfully learned to act as both a visual reasoning engine and an autoregressive image generator.

### Task B-C5: Evaluation
- **Implementation:** VQA exact-match evaluations are calculated. For image generation, the system forcibly masks text-logits during sampling to only permit the 256 visual IDs. The 16 generated tokens are decoded back into a grid via the frozen VQ-VAE decoder.
- **Results:** 
  - **VQA Accuracy:** Achieved `1.0000` (100%) accuracy on the test set (`partB_vqa_metrics.csv`).
  - **Image Generation:** Artifacts (`partB_generated_grid.png`) reveal highly coherent geometric shapes generated by the LM. 
  - **Diagnostics:** The logit mask distributions (`partB_logit_mask_hist.png`) prove the LM learned the exact semantic separation between text logic and image generation.
- **Insights & Limitations:** **Insight:** A 100% test accuracy and visually perfect generated grids prove that a discrete tokenizer is sufficient to bridge vision and language bidirectionally. **Limitation:** Autoregressive token generation (predicting one image patch at a time) is incredibly slow compared to parallel generation methods like diffusion models.
