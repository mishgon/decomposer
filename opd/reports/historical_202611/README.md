# Historical OPD Report Scripts

Preserved November 2026 report-generation code, previously kept only in `tmp/pdfs`.
These scripts contain historical run-specific captions and assumptions. Review and
update those before using them for another run; they are not live dashboards.

Run from the repository root. Inputs remain external JSON snapshots under
`tmp/pdfs/`; the input filenames are named at the top of each script. Create
`output/pdf/` before running. Dependencies: numpy, matplotlib, reportlab, and
PyMuPDF (for the HF token-budget plot). Raw snapshots and generated PDFs are not
included in Git. Syntax was checked; the reports were not regenerated during archival.
