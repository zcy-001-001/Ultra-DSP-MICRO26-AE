# Table 7 result provenance

The formal AE files in this directory are:

- `table7_paper_anchor.json`
- `table7_paper_anchor.csv`
- `table7_paper_anchor.md`

They are copied paper anchors and use `source_kind=PAPER_ANCHOR`. This AE
package does not claim a newly generated Table 7 result.

The older `mixed_precision_*`, `raw_logs_sanitized/`, and
`table7_geomean_*` files are retained as non-canonical development evidence so
that prior work is not silently deleted. They must not be cited as the formal
Table 7 result. A future evaluator run should use the
`table7_optional_recomputed_*` output names documented by the parent README.
