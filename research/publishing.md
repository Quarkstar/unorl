# Build and publish the book

From the repository root, install the Python plotting dependencies and the locked
Node dependency, then generate the book:

```bash
python -m venv .venv-docs
.venv-docs/bin/pip install -r requirements-docs.txt
npm ci
.venv-docs/bin/python scripts/research/build.py
npm run docs:build
python -m http.server 8000 --directory _build/html
```

The theme is MyST `book-theme`, with a table of contents, experiment chapters,
search and downloadable data. Plot colors use the Google Material 500 palette;
SVG is used in the book and PNG copies are provided for presentations.

## GitHub Pages

Push this repository to GitHub, then choose **Settings → Pages → Source → GitHub
Actions**. The `pages.yml` workflow builds the book from committed snapshots and
deploys on pushes to `main` or a manual workflow dispatch. Pull requests build and
check the book without deploying. There is no GPU requirement for either workflow.

The deployment obtains the site's base path from GitHub's Pages configuration.
For a repository named `unorl`, the normal URL is `https://OWNER.github.io/unorl/`.
For `OWNER.github.io` repositories or custom domains, GitHub's configured base path
is used. The workflow does not contain a hard-coded owner or repository URL.

This project's repository is [Quarkstar/unorl](https://github.com/Quarkstar/unorl),
and the published book is [quarkstar.github.io/unorl](https://quarkstar.github.io/unorl/).

See the [official MyST GitHub Pages guide](https://mystmd.org/guide/deployment-github-pages)
for the Pages source setting and base-path behavior.

## Updating records

The contribution guide is
`CONTRIBUTING.md` at the repository root. Add a curated record to
`scripts/research/snapshot.py`, export from local `runs/`, then regenerate with
`scripts/research/build.py`. Review the emitted JSON, figures and experiment page
before committing. A fresh clone only needs the committed snapshots to rebuild.
