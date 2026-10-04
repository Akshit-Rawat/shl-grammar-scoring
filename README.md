# Grammar Scoring Engine for Spoken English

Predicts a continuous grammar score (1–5, mean opinion score) from a 45–60 second recording of spoken English.
Built for the **SHL Hiring Assessment 2026** Kaggle challenge (769 training clips, 216 test clips).

The full solution, with documented code, visualisations and a written report, is in
[`notebookb802b42a79.ipynb`](notebookb802b42a79.ipynb).

## Results

| | RMSE | Pearson | R² |
|---|---|---|---|
| Baseline (always predict the training mean) | 1.014 | – | 0.00 |
| **Training RMSE** (final model on all 732 training clips) | **0.333** | 0.949 | 0.89 |
| **5-fold cross-validation** (realistic estimate) | **0.535** | **0.850** | 0.72 |
| **Public leaderboard** | **0.3846** | | |

The model halves the baseline error and explains about 72 % of the variance in grammar scores on unseen clips.

## Approach

Grammar shows up in **what** a speaker says (word choice, sentence structure) and in **how** they say it
(hesitation, fluency). Each recording is therefore described in two complementary ways and combined in a
regularised linear model.

```
                ┌─► Whisper small.en transcription (prompted to keep fillers and mistakes)
                │        ├─► CoLA grammar model (RoBERTa): sentence acceptability scores + embedding
                │        ├─► LanguageTool: grammar errors per word
                │        ├─► fluency & complexity: speech rate, fillers, repetitions, sentence length, vocabulary
                │        └─► sentence embedding (all-mpnet-base-v2)
audio (16 kHz) ─┤
                ├─► wav2vec2-base embeddings (layer 8, whole clip, mean + std over time)
                └─► Whisper-small encoder embeddings (layer 9 chosen by CV, 20 s chunks, mean + std)

4,629 features → standardise → equal weight per feature block → Ridge regression → score clipped to [1, 5]
```

**Key preprocessing decisions**
- Audio resampled to 16 kHz mono, leading/trailing silence trimmed, peak-normalised.
- **37 training clips scored 0 were excluded.** A score of 0 is not on the 1–5 rubric, these clips form one
  separate batch (consecutive IDs, 30 of 37 exactly 60.0–60.1 s, some with no intelligible speech), and no test
  clip resembles them. Training uses the remaining 732 clips.
- Feature blocks of very different sizes (21 to 1,536 features) are each scaled by 1/√(block size), so no block
  dominates the Ridge penalty.

**Evaluation:** stratified 5-fold cross-validation. Ridge's `alpha` is chosen inside each training fold,
so out-of-fold scores are honest.

## What each part contributes (ablation, CV RMSE on 732 clips)

| Features | CV RMSE |
|---|---|
| Sentence embedding only | 0.778 |
| Interpretable fluency + grammar features (21) | 0.727 |
| CoLA grammar embedding only | 0.644 |
| wav2vec2 audio only | 0.606 |
| Whisper encoder audio only | 0.555 |
| Text features + Whisper audio | 0.547 |
| **All blocks (final model)** | **0.535** |

**Findings**
- Audio carries most of the signal, and the Whisper encoder is the strongest single block.
- Transcript features add complementary grammar information.
- A grammar-tuned text representation (CoLA) is far more useful than a general-purpose sentence embedding.
- Remaining error is concentrated at the extremes: clips scored 2 are over-predicted (≈ 2.6 on average) and
  clips scored 5 under-predicted (≈ 4.6), i.e. predictions are pulled towards the middle of the scale.

## Development history (public leaderboard RMSE)

| Model | Score |
|---|---|
| wav2vec2 audio embeddings + Ridge | 0.4742 |
| Whisper encoder embeddings + SVR | 0.4722 |
| + transcripts and grammar features | 0.4144 |
| **+ Whisper encoder block, score-0 clips removed (final)** | **0.3846** |


## How to run

1. Create a Kaggle notebook with the competition data attached and import `notebookb802b42a79.ipynb`.
2. Settings: **Accelerator = GPU**, **Internet = On** (pretrained models are downloaded from Hugging Face).
3. *Run All* (about 1–1.5 hours, with transcription as the slowest step). Outputs are written to `/kaggle/working`:
   `submission.csv`, `results.json`, `ablation.csv`, `oof_predictions.csv`, transcripts and the fitted model.

Python packages are listed in `requirements.txt`. LanguageTool also needs Java, which is available on Kaggle.

## Repository contents

| Path | Description |
|---|---|
| `notebookb802b42a79.ipynb` | Final notebook: code, outputs, visualisations, report, training RMSE |
| `submission.csv` | Test-set predictions submitted to Kaggle (216 rows: `filename`, `label`) |
| `results/results.json` | Configuration and metrics of the final model |
| `results/ablation.csv` | Cross-validated scores of every feature combination |
| `results/oof_predictions.csv` | Out-of-fold prediction for each training clip |
| `requirements.txt` | Python dependencies |

## Limitations

- Text features depend on transcription quality. Mistakes that Whisper mishears or silently corrects are invisible to them.
- Pooling embeddings over time discards word order in the audio features.
- CoLA was built from written sentences, so its judgements are noisy for informal speech.
- Labels are averages of human ratings, so some disagreement is inherent.
