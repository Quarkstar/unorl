# Contributing to UNORL

Keep algorithm changes separate from optimizer, adapter and data changes. A run
comparison must state response count, prompt exposure, trainable parameters,
optimizer/LR, loss reduction, KL, masking, generation settings and evaluation size.

## Checks

```bash
python -m pip install -e '.[dev]' -r requirements-docs.txt
ruff check unorl scripts tests
ruff format --check unorl scripts tests
pytest tests/test_repository.py
python scripts/research/build.py
npm ci
npm run docs:build
```

`tests/test_low_resource.py` and `tests/test_ppo_warmup.py` additionally require the
pinned SkyRL training environment, torch, PEFT and transformers. Run them in that
environment with the SkyRL checkout and this repository on `PYTHONPATH`.

## Recording experiments

1. Save the run's immutable config, code revision, metrics and data audit in `runs/`.
2. Add the run's interpretation/group to `scripts/research/snapshot.py` and export
   with `python scripts/research/snapshot.py`. This is the only step needing local
   run files. Never overwrite original logs to change their names or interpretation.
3. Run `python scripts/research/build.py` to regenerate all pages, plots, comparison
   CSV and the book navigation. Generated pages should be edited through the
   snapshot metadata or generator, not manually.
4. Review source hashes, completeness, optimizer confounds and missing metrics.
   Commit `research/data/`, pages and figures together. Never commit weights,
   caches, solution dumps, credentials or machine-specific asset paths.

Documentation-only builds work without GPU dependencies. The documentation tests
check that the published evaluation sample counts agree with the summary metrics.
