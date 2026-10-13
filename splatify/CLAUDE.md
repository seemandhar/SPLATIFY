# Splatify — Claude Multi-Agent Pipeline for Gaussian Splatting

This project converts Gaussian Splatting research papers into complete nerfstudio
method implementations using the gs_template framework and specialized Claude subagents.

## Architecture

The orchestrator agent coordinates 8 specialized subagents:

1. **parser** — Extracts PDF to markdown, cleans it for GS implementation
2. **planner** — Creates architecture DAG, maps paper to gs_template extension points
3. **coder** — Generates complete gs_template method from paper + templates + 20 examples
4. **reviewer** — Reviews code for errors and API compliance with SplatfactoModel
5. **validator** — Checks all paper novelties/equations are implemented
6. **integrator** — Validates cross-file consistency and MethodSpecification wiring
7. **tester** — Runs smoke tests in nerfstudio conda environment (gsplat rasterization)
8. **debugger** — Diagnoses and fixes errors (gsplat, CUDA, nerfstudio-specific)

## Pipeline Flow

```
parser → planner → coder → reviewer (loop) → validator → integrator → tester (debug loop) → train
```

## Target Framework: gs_template

The gs_template framework at `base-code/gs_template` provides:

- **GSTemplateModel** — Base class extending SplatfactoModel with 9 extension points
- **rendering.py** — Shared helpers: prepare_camera, rasterize_gaussians, assemble_outputs
- **losses.py** — 30+ reusable loss functions (depth, feature, regularization, etc.)
- **datamanagers.py** — Custom DataManagers (depth estimation, feature extraction)
- **shading.py** — SH evaluation, normals, cubemap sampling
- **optimizers.py** — Custom optimizers (BOGausSAdam)
- **configs.py** — MethodSpecification registry for nerfstudio entry points
- **methods/** — 20+ implemented papers as in-context examples

## Extension Points (9 total)

1. `populate_modules()` — Add new per-Gaussian parameters, MLPs, state
2. `get_outputs()` — Custom rendering using gsplat rasterization
3. `compute_custom_losses()` — Novel loss terms
4. `split_gaussians()` — Handle new params during densification
5. `get_gaussian_param_groups()` — Register Gaussian param optimizers
6. `get_param_groups()` — Register non-Gaussian params (MLPs)
7. `get_outputs_for_camera()` — Custom viewer/eval outputs
8. `load_state_dict()` — Checkpoint loading for new params
9. `get_metrics_dict()` — Custom evaluation metrics

## Complexity Tiers

- **Tier 1** (Config-only): Just change splatfacto defaults
- **Tier 2** (Custom loss): Override compute_custom_losses() only
- **Tier 3** (Loss + rendering): Override get_outputs() + losses
- **Tier 4** (All extension points): New params, rendering, loss, densification
- **Tier 5** (Advanced): Fundamental architecture changes

## Key Directories

- `workspace/` — Per-run working files (cleaned paper, plans, results)
- `generated/` — Per-run generated implementations
- `agents/` — Agent definitions and system prompts

## Required Output Files

Every generated implementation must contain a Python package directory named after the
method (method_name_snake from the DAG plan, e.g., `dashgaussian/`). All internal imports
use this package name. Example for a method named "dashgaussian":
- `dashgaussian/__init__.py` — Package init
- `dashgaussian/model.py` — Model + Config class
- `dashgaussian/losses.py` — Paper-specific losses
- `dashgaussian/rendering.py` — Custom rendering if needed
- `dashgaussian/configs.py` — MethodSpecification for nerfstudio
- `dashgaussian/datamanager.py` — Custom DataManager if needed
- `README.md` — Documentation
- `pyproject.toml` — Package with nerfstudio entry points (referencing `dashgaussian.configs:...`)

## Nerfstudio Environment

All testing/training commands must run in the `nerfstudio` conda environment:
```bash
eval "$(conda shell.bash hook)"; conda activate nerfstudio; <command>
```

## Usage

```bash
# CLI
python main.py --arxiv 2308.12345
python main.py --pdf /path/to/paper.pdf --method-name my_gs

# Web UI
python web.py  # http://localhost:5006
```
