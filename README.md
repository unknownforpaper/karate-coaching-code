# Karate Coaching: Classification, DTW Correction and RAG Feedback

Code and data for the double-blind submission "AI-Powered Karate Coaching:
Technique Classification, DTW Pose Correction and Retrieval-Augmented Feedback,
with a Negative Result for Dual-Stem Fusion".

## Contents

- `02_Master_Pipeline_anonymized.ipynb`: the full pipeline. It covers the CV and
  golden-split comparison of the four classifiers, the 11-clip Academy test,
  the OOD cascade, DTW correction and the feedback ablation.
- `backend/`: helper modules imported by the notebook (DTW, LLM client,
  coaching-chunk store).
- `.env.example`: lists the optional `GROQ_API_KEY`.

## Data

All data is hosted at: **[PASTE OSF VIEW-ONLY LINK]**

Download it and place it so the folders look like this:

```
data/
  processed/karate_normalized_base.csv   # 102-feature landmark data, 30-frame windows
  raw_videos/
    references/         # 3 reference executions used for DTW
    academy_trainee/    # 11 clips, practitioners 048-051 (suffix 001 = correct, 002 = flawed)
    distractors/        # 7 non-karate clips (JumpJacks, Squats, one NTU RGB+D clip)
```

The 484 tutorial clips used for training were collected from public web videos
and are not redistributed. The CSV contains the extracted landmark features
only. [Add: "A list of source IDs is in `data/processed/<file>`" if you include one.]

## How to run

1. Python 3.12 is recommended (the notebook was developed on 3.12.7).
2. Install the dependencies: [TensorFlow/Keras, XGBoost, scikit-learn, MediaPipe,
   pandas, NumPy, SciPy, matplotlib, python-dotenv, plus whatever `backend/`
   needs. List them or add a `requirements.txt`.]
3. Put the data in `data/` as shown above.
4. Open the notebook and run the cells in order. The paths in the first
   configuration cell are relative to the repository root.

## Notes for reviewers

- **Randomness.** The seed is 42, but the neural networks are not bit-for-bit
  reproducible between executions (see Sec. V-A of the paper). The tree models
  reproduce exactly, so expect different Bi-LSTM numbers on a re-run.
- **LLM stage (optional).** The coaching-feedback cells call an LLM through Groq.
  Without a key, the notebook uses an offline fallback and still runs; only the
  live-LLM arms of the ablation differ. To use it, copy `.env.example` to
  `backend/.env` and add your own `GROQ_API_KEY`.
- **Run time.** [Add rough time, and whether a GPU is needed.]
- **Outputs.** The notebook writes result tables (e.g. `cv_results.csv`,
  `ood_scores.csv`, `feedback_ablation.csv`) used in the paper.# karate-coaching-code
