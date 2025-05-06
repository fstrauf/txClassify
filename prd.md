# Merchant Name Cleanup and Embedding Enhancement Plan

## 🎯 Objective

Implement Named Entity Recognition (NER) to clean and standardize merchant or transaction descriptions using spaCy or Flair, before passing them into the embedding pipeline for categorization. Optionally explore stacked embeddings to improve cosine similarity accuracy.

---

## 📦 Existing Pipeline (Before)

1. Incoming transaction descriptions via API (`/train` or `/classify`)
2. Clean descriptions using regex and heuristics (`clean_text`)
3. Run embeddings via **Replicate** (`beautyyuyanli/multilingual-e5-large`)
4. Store embeddings
5. Use **cosine similarity** to match to training data

---

## 🧠 Planned Improvements

### 1. Add NER for Entity Extraction

#### Why:

- Extract cleaner merchant names (`ORG`, `PRODUCT`, `SERVICE`, etc.)
- Reduce noise and inconsistency before embedding step

#### Options:

- ✅ **spaCy**: Lightweight, easy to deploy, fast
- ✅ **Flair**: Better on noisy or informal text, slower, better recall

#### Process:

- Run NER on each transaction description
- Extract entity labeled `ORG` or similar (fallback to original if missing)
- Use this cleaned version for embedding

---

### 2. (Optional) Use Stacked Embeddings

#### Idea:

- If training data contains many items per category, compute **centroid embeddings per category**
- Compare new embedding not to all training samples, but to **category centroids**
- Can improve categorization speed and potentially accuracy

#### Stack Options:

- Flair supports stacking embeddings (GloVe + BERT + Flair)
- spaCy supports transformer-based embeddings via `spacy-transformers`

---

## 🧱 Implementation Steps

### Step 1: Add NER Module (e.g. `ner_utils.py`)

- Load spaCy (`en_core_web_md`) or Flair model
- Define `extract_merchant_name(text: str) -> str`
- Fall back to original description if no entity found
- Integrate after `clean_text()` but before embedding generation

### Step 2: Evaluate NER Accuracy

- Run batch of descriptions through spaCy and/or Flair
- Compare extracted entities to expected merchant names

### Step 3: Add `use_ner` Flag to `/train` and `/classify`

- Toggle NER on/off for A/B testing
- Optional CLI debug command for single input

### Step 4: (Optional) Category Embedding Stacking

- For each category, average embeddings from training samples
- Store these centroids in DB or in-memory cache
- During classification, compare new embedding to category centroids
- Use cosine similarity for scoring

---

## 🧪 Hosting Considerations

### spaCy

- Can be embedded directly in your Flask app
- Light memory (~100–500MB), CPU-friendly
- Good for Docker container setup (no GPU needed)

### Flair

- Needs more memory (~1GB+)
- Slower inference; consider async background worker or batch processing
- Best run on minimal GPU if stacking embeddings

---

## 🐳 Deployment Plan

- Package NER as standalone Python module
- Integrate into existing Flask app (`@require_api_key` protected endpoints)
- Build minimal Docker container:
  ```Dockerfile
  FROM python:3.11-slim
  RUN pip install spacy flask
  RUN python -m spacy download en_core_web_md
  COPY . /app
  WORKDIR /app
  CMD ["python", "app.py"]
  ```
