# Copying this folder to GitHub

Use the **contents of `NCV_github/` as the repository root** for
[`JZHANG4362/NCV`](https://github.com/JZHANG4362/NCV), so `README.md`,
`reproduce.py`, `requirements.txt`, and the six section folders are at the top
level. The accompanying ZIP contains the same prepared files in a single
`NCV_github/` directory.

Include `.gitignore`, the 98 saved result files, the shared raw datasets,
templates, and the `reference/` directory. Generated `outputs/`, `runs/`, plot
intermediates, caches, and local logs are excluded. The largest input is the
approximately 11.5 MB MNIST archive.

If the destination repository already has files, review and merge the changes
there; this preparation did not inspect or alter its remote contents. A
software `LICENSE` has not been chosen. Retain the dataset attribution notes
and the openly documented replication limitations.

Validate a clean checkout with:

```bash
python scripts/check_package.py
python reproduce.py archive
```

No Git repository was initialized and nothing was pushed or published during
preparation.
