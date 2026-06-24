# Data and privacy

This public repository intentionally excludes:

- raw Human-AI conversations;
- user-level or turn-level text exports;
- manually reviewed samples containing conversation text;
- full prediction datasets;
- audio recordings and meeting transcripts;
- trained model binaries;
- upstream cVAE embeddings and latent-score files.

Only aggregate statistics, scripts, documentation, and figures are included.

The private data pipeline joins files using `conversation_hash`. Researchers
with authorized access should set `GENAI_DATA_ROOT` to the local organized data
directory before running the scripts.

