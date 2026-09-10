#!/usr/bin/env bash
# Full reproduction. CPU only. ~15 min on one core.
set -euo pipefail
pip install -r requirements.txt
python -m arcstate.tasks                                   # generator self-test
# runs/main.pt is committed. To retrain from scratch:
# python -m arcstate.train --steps 9000 --bs 24 --lr 3e-3 --ops identity --tag main
python -m arcstate.experiments --ckpt runs/main.pt         # -> web/results.json
python tools/export.py runs/main.pt                        # -> web/model.json (+ parity fixture)
python tools/build.py                                      # -> state-surgery.html
python tools/make_pdf.py docs/one-page-summary.pdf          # one-page concept summary
echo "Open state-surgery.html. The header must read 'engine verified vs PyTorch'."
