# paper-lint

Hygiene tooling for our paper drafts. Runs a third-party AI-text detector (Hello-SimpleAI/chatgpt-detector-roberta, trained on HC3) over the draft via GitHub Actions and reports per-section scores. No paper content lives in this repo; the draft is fetched from its gist URL at run time.

Detectors are directional, not verdicts, especially on technical prose. Use the output to find robotic sections, not to certify anything.
