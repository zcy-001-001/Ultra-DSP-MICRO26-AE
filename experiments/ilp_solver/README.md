# ILP Solver Notebooks

This folder contains three standalone Jupyter notebooks for layout search and plotting. Each notebook can be run independently from a clean Python kernel.

## Files

- `pareto.ipynb`: defines the layout solver and renders Pareto-frontier plots for selected mixed-precision cases.
- `parallelism.ipynb`: computes packing grids and renders computational-parallelism surfaces for single-device and cross-device configurations.
- `efficiency.ipynb`: computes computational-efficiency-ratio heatmaps and the INT16/INT32 MAC generality comparison.

## How to Run

Open any notebook in Jupyter and run all cells from top to bottom.

For a command-line check without modifying the source notebooks, run:

```bash
jupyter nbconvert --to notebook --execute pareto.ipynb --output pareto.executed.ipynb
jupyter nbconvert --to notebook --execute parallelism.ipynb --output parallelism.executed.ipynb
jupyter nbconvert --to notebook --execute efficiency.ipynb --output efficiency.executed.ipynb
```

The notebooks display figures inline and do not save figures to disk.

## Dependencies

The notebooks require Python 3 with `numpy`, `scipy`, `matplotlib`, and `jupyter`. The optional `adjustText` package improves plot-label placement; when it is unavailable, the notebooks use a no-op fallback.
