# SPLATIFY

Paper → Code → Discover. A Claude agent reads a 3DGS paper and writes the implementation, then 8 specialized agents compete across generations to discover a novel method that beats the baseline.

## Demo

Live demo: [demo.html](./demo.html)

- **Act 1** — paper2code: the agent reads the 3D Gaussian Splatting paper (Kerbl et al. 2023), recovers citations, plans the architecture, writes the `gs_template` implementation, renders preview frames, spots artifacts, and fixes the code from a visual analysis.
- **Act 2** — knowledge discovery arena: 8 island agents compete across 3 generations with cross-generation knowledge transfer, converging on a novel method (**NebulaGS**) that beats the baseline by +3.42 dB.

## Run locally

```bash
python3 -m http.server 8000
# visit http://localhost:8000
```
