# Overview of Proposed Architectures for VAD Scoring Models

This document outlines two primary ways to categorize the architectures used for Valence–Arousal–Dominance (VAD) scoring models in the project.  
The separation can be made based on either the **encoder design** or the **prediction target**.

---

## 1. Encoder-Based Separation
Two types of encoders are considered in the current work:

### 1.1 MelEncoder
- Adapted from the GPT-SoVITS pipeline.
- Produced embedding is used to guide audio generation.
- Presumably contains information about the speaker’s emotional state.
- Experimental observations:  
  Zeroing out certain embedding values preserved the speaker’s identity in generated speech but changed the emotional tone, suggesting that emotional information is embedded in specific dimensions.
- Likely **does not contain significant semantic content** (the meaning of words).  
  This conclusion comes from observing that when the **EmoWhisper** encoder is used, audio containing the phrase *"Thank you"* is consistently classified as belonging to the *Thankful* emotion distribution. This suggests that semantic information in EmoWhisper embeddings influences classification.

### 1.2 EmoWhisper
- Uses the last hidden state of a modified Whisper model.
- This representation **definitely** contains emotional information.
- However, since Whisper was originally trained as an ASR model, it is optimized for extracting **semantic content** (text meaning) from audio.
- This can lead to **semantic leaking**—a phenomenon where semantic content inadvertently influences emotion classification.

### 1.3 Emotion2Vec-base
- Good backbone with clear and strong output. The model is showing a good internal clasterization ability.
- Seems to be intacked by the semantics.

---

## 2. Phenomenon: Semantic Leaking
**Definition:**  
Semantic leaking occurs when semantic content (the *meaning* of words) influences a model’s emotion classification, even if the emotional tone (acoustic features) differs.

**Speech Components:**
1. **Semantic Component** – The meaning of the spoken words, which can be transcribed into text.
2. **Acoustic Component** – How the words are spoken (pitch, intonation, rhythm, etc.). Emotional content is part of this component.

**Interaction Between Components:**
- The acoustic component can influence the perceived meaning of the words (e.g., *"Thank you"* can be sincere or sarcastic).
- Conversely, if an encoder captures strong semantic features, a classification head might prioritize them over acoustic features, potentially misclassifying emotional tone.

**Relevance for VAD Scoring:**
- For VAD models, designed for text-to-speech systems, **acoustic features should be prioritized** over semantic ones since they carry the emotional signal.
- Excessive semantic influence (as seen in semantic leaking) is **undesirable** and should be mitigated.

---

## 3. Prediction Value–Based Separation
Models can differ in how they predict VAD values:

### 3.1 Means-Only Prediction
- The model predicts **only** the VAD mean values (central tendency of the distribution).
- Simple, direct, and effective for many cases.

### 3.2 Means + Standard Deviations Prediction
- The model predicts both:
  - **Means** of the VAD distributions.
  - **Standard deviations (stds)**, representing the spread of the distribution for the given sample.
- Motivation:
  - Emotions are not categorical but continuous.
  - Similar to a Variational Autoencoder (VAE), where:
    - Means and stds define a latent distribution.
    - The latent space here is the VAD space, representing emotional states.
- Benefits:
  - Captures uncertainty and variability in emotional expression.
  - Enables probabilistic modeling of emotional states rather than fixed-point estimates.

---

## 4. Summary
Architectures for VAD scoring in this project can be categorized along two dimensions:
1. **Encoder choice** – MelEncoder (acoustic-focused) vs. EmoWhisper (semantic + acoustic, prone to semantic leaking).
2. **Prediction target** – Means only vs. Means + Stds.

In designing these models, care must be taken to **limit semantic leaking** and ensure that **acoustic cues**—the primary carriers of emotional information—are emphasized in training and inference.
