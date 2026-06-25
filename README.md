# VIPS: Vehicle-Infrastructure Cooperative Planning Benchmark via Pseudo-Simulation

Official code for **VIPS** (ECCV 2026).

Hoonhee Cho, Jae-Young Kang, Giwon Lee, Hyemin Yang, Heejun Park, Kuk-jin Yoon<br>
KAIST, Visual Intelligence Lab<br>
Contact: {gnsgnsgml, mickeykang, dlrldnjs, hyemin0806, parkhee.ticket, kjyoon}@kaist.ac.kr

_Paper / project page / arXiv: coming soon._

VIPS evaluates **vehicle-infrastructure cooperative (V2X) planning** on the
[V2X-Real](https://mobility-lab.seas.ucla.edu/v2x-real/) dataset with a
**two-stage pseudo-simulation** EPDMS (Extended Predictive Driver Model Score),
adapting the NAVSIM v2 closed-loop protocol to the cooperative V2X setting.
This repository is the official VIPS evaluation framework: V2X-Real metric-cache
generation, scene-parallel temporal evaluation with selectable background-traffic
policies (log-replay / IDM / constant-velocity), and EPDMS scoring over the
benchmark scene subset that the reported numbers use.

## Highlights

- **Two-stage pseudo-simulation EPDMS** on V2X-Real (stage 1: original ego
  pose; stage 2: offset poses; Gaussian-kernel combined score).
- **Selectable background-traffic policies**: `log_replay` (ground-truth replay),
  `idm` (reactive Intelligent Driver Model on V2X-Real lanes), and
  `constant_velocity`.
- **Scene-parallel temporal evaluation** across multiple GPUs/workers.
- **Benchmark scene subset** scored directly at eval time for the reported numbers.
- **Agents**: `cos_v2x` (the CoS-V2X model), `admlp`, the `constant_velocity` /
  `stop` / `human` baselines, and custom planners via `AGENT_CLASS_PATH`.

## Repository layout

```text
navsim/
  agents/                      # constant_velocity, stop, human, cos_v2x, admlp_sim_v2
  common/                      # dataclasses + V2X-Real dataloaders
  evaluate/                    # EPDMS entry points
  planning/
    metric_caching/            # V2X-Real two-stage metric cache generation
    script/                    # run_pdm_score_v2xreal_stage2_temporal[_parallel].py
    simulation/                # PDM planner, scorer, IDM traffic observation
    training/                  
  traffic_agents_policies/     # log_replay / constant_velocity / IDM policies
  visualization/               # BEV + stage-2 evaluation rendering
scripts/
  data/                        # HuggingFace asset download helper
  evaluation/                  # eval + metric-cache entry shells, config helper
```

## Installation

```bash
git clone https://github.com/mickeykang16/VIPS.git
cd VIPS
conda env create -f environment.yml   # creates the `vips` env
conda activate vips
pip install -e .
```

The `vips` env covers metric-cache generation and the `constant_velocity` /
`stop` / `human` baselines. The **CoS-V2X** model and the **ADMLP** planner run in
CoS-V2X's separate `mmcv` / `mmdet3d` environment — see
[Evaluation → CoS-V2X](#cos-v2x).

## Data preparation

This repo does not include V2X-Real data, generated novel-view images, maps,
metric caches, experiment outputs, or model checkpoints.

Pick an evaluation root that everything below assembles into:

```bash
export V2XREAL_ROOT=/path/to/V2X-Real-eval
```

**1. Official V2X-Real raw test images.** Download the
`OPV2V format (V2X-Real-Lidar-Cameras)` `Test_set` from the
[UCLA V2X-Real release page](https://mobility-lab.seas.ucla.edu/v2x-real/)
([current Box link](https://ucla.app.box.com/s/x9k5nyt4szrxbtnn8xk8sqiuhg5noqgv))
and place it at `$V2XREAL_ROOT/data/test`.

**2. Generated VIPS evaluation assets** (novel-view images, map, frame infos),
distributed via a public HuggingFace Dataset repo:

```bash
export HF_ASSET_DIR=/path/to/v2xreal_eval_assets   # local dir to download the assets into
# needs a recent huggingface_hub for the `hf` CLI:  pip install -U huggingface_hub
hf download mickeykang/VIPS-v2xreal-assets --repo-type dataset --local-dir "$HF_ASSET_DIR"
```

Expected asset layout:

```text
/path/to/v2xreal_eval_assets/
  meta/
    spd_infos_temporal_test.pkl
    maps_final/v2x_real_map.json
  archives/
    test_novel.tar.zst.part-*
    test_novel_infra.tar.zst.part-*
    SHA256SUMS
```

Restore the generated assets into your evaluation root (or use the helper script
`scripts/data/download_v2xreal_eval_assets.sh`):

```bash
# V2XREAL_ROOT was set above; HF_ASSET_DIR was set when downloading the assets
(cd "$HF_ASSET_DIR/archives" && sha256sum -c SHA256SUMS)
mkdir -p "$V2XREAL_ROOT/data" "$V2XREAL_ROOT/infos/v2xreal/cooperative"
cat "$HF_ASSET_DIR"/archives/test_novel.tar.zst.part-*       | tar --zstd -xf - -C "$V2XREAL_ROOT/data"
cat "$HF_ASSET_DIR"/archives/test_novel_infra.tar.zst.part-* | tar --zstd -xf - -C "$V2XREAL_ROOT/data"
cp -r "$HF_ASSET_DIR/meta/maps_final" "$V2XREAL_ROOT/"
cp "$HF_ASSET_DIR/meta/spd_infos_temporal_test.pkl" "$V2XREAL_ROOT/infos/v2xreal/cooperative/"
```

Final layout and sanity check:

```text
$V2XREAL_ROOT/
  data/{test, test_novel, test_novel_infra}/
  maps_final/v2x_real_map.json
  infos/v2xreal/cooperative/spd_infos_temporal_test.pkl
```

```bash
cp configs/eval/paths.example.py configs/eval/paths.py   # then edit your paths in it

# sanity check — the paths you set should resolve:
eval "$(python3 scripts/evaluation/eval_config.py configs/eval/paths.py)"
test -f "$V2XREAL_PKL_PATH" \
  && test -f "$V2XREAL_MAP_ROOT/v2x_real_map.json" \
  && test -d "$SENSOR_BLOB_PATH/test" && echo "paths OK"
```

## Metric cache

```bash
conda activate vips
# builds both stages in one go; paths come from configs/eval/paths.py
NUM_CPUS=16 bash scripts/evaluation/run_metric_caching_v2xreal_stage2_coop_5s.sh
```

## Evaluation

All evaluation goes through one entry script
(`scripts/evaluation/run_cv_pdm_score_evaluation_v2xreal_stage2_coop_temporal_parallel.sh`).
Activate the conda env your chosen config needs, then pass the config with
`CONFIG=` (anything set on the command line overrides the config). Concrete
commands for each agent are below.

### Experiment configs

Each experiment is a small python config under [`configs/eval/`](configs/eval) —
`cos_v2x.py`, `admlp.py`, `constant_velocity.py`, `human.py`, `stop.py` — holding
the agent, the conda env it needs (`CONDA_ENV`), traffic policy, GPUs and workers
(plus the model paths for `cos_v2x` / `admlp`). Only the machine-specific
V2X-Real **data** paths live in `configs/eval/paths.py`.

### Background-traffic policy

The background-traffic policy is set by `TRAFFIC_POLICY` in the experiment config
(`configs/eval/*.py`). **The reported VIPS benchmark numbers use `log_replay`**,
which is the default; `idm` and `constant_velocity` are optional alternatives —
change `TRAFFIC_POLICY` in the config to use them (or override it for a single run
on the command line, e.g. `TRAFFIC_POLICY=idm bash ...`).

| `TRAFFIC_POLICY` | Behavior |
|---|---|
| `log_replay` (default — used for the reported numbers) | Ground-truth tracks replayed from the metric cache |
| `idm` (optional) | Reactive Intelligent Driver Model on V2X-Real lanes |
| `constant_velocity` (optional) | Detected agents extrapolated at constant velocity |

### CoS-V2X

CoS-V2X has its own `mmcv` / `mmdet3d` dependencies. Clone it into `models/CoS-V2X`,
set up its conda environment, and install **this** repo into that same env so the
`cos_v2x` agent can import both NAVSIM and the model:

```bash
git clone https://github.com/mickeykang16/CoS-V2X models/CoS-V2X
conda activate <cos-v2x-env>   # the env created per models/CoS-V2X/README.md
pip install -e .               # install VIPS alongside the model
```

`configs/eval/cos_v2x.py` already points at `models/CoS-V2X` (folder / config /
checkpoint — `COS_V2X_FOLDER` / `COS_V2X_CONFIG_PATH` / `COS_V2X_MODEL_CHECKPOINT_PATH`).
Download the checkpoint per the CoS-V2X README, then run:

```bash
conda activate <cos-v2x-env>
CONFIG=configs/eval/cos_v2x.py EXPERIMENT_NAME=cos_v2x_coop \
  bash scripts/evaluation/run_cv_pdm_score_evaluation_v2xreal_stage2_coop_temporal_parallel.sh
```

### Non-model baselines

```bash
conda activate vips
CONFIG=configs/eval/constant_velocity.py \
  bash scripts/evaluation/run_cv_pdm_score_evaluation_v2xreal_stage2_coop_temporal_parallel.sh
```

`constant_velocity`, `stop`, and `human` run in the `vips` env — swap the config
(`configs/eval/{stop,human}.py`). `human` replays the ground-truth future
trajectory. `admlp` is a custom planner
(`configs/eval/admlp.py`, `AGENT_CLASS_PATH=navsim.agents.admlp_sim_v2.ADMLPSim`)
and runs in the CoS-V2X env.

### Benchmark scene subset

The benchmark scores a fixed **no-stop** subset
(`scripts/evaluation/test_scene_tokens_5s_no_stop.txt`; 10 scenes, 643 stage-1 /
180 stage-2 tokens) — scenes where the ego is actually driving, since stationary
scenes are degenerate for planning (a do-nothing policy scores well). It is
applied by default, so the printed score is the reported number; set
`SCENE_FILTER_FILE=""` for the full test split.

## Results

**EPDMS** as reported in the VIPS paper. The headline numbers use the
`log_replay` traffic policy; `idm` is the optional reactive-traffic variant.

| Method | Log replay | IDM |
|---|---|---|
| Constant Velocity | 5.88 | 10.36 |
| AD-MLP | 32.31 | 31.97 |
| MomAD | 45.38 | 45.82 |
| Uni-V2X | 43.79 | 44.33 |
| **CoS-V2X** | **50.88** | **51.00** |


## Visualization

Set `VISUALIZE=1` in the eval (config or command line) to save per-token BEV
renders — stage-1/stage-2 proposals, EPDMS sub-scores, and the drivable area —
under `<output_dir>/viz/` (via `navsim/visualization/stage2_eval_viz.py`).

## License

Released under the Apache License 2.0 (see [LICENSE](LICENSE)). This code builds
on [NAVSIM](https://github.com/autonomousvision/navsim) and the
[nuPlan devkit](https://github.com/motional/nuplan-devkit). The V2X-Real data is
subject to the original UCLA V2X-Real terms.

## Citation

```bibtex
@inproceedings{cho2026vips,
  title     = {{VIPS}: Vehicle-Infrastructure Cooperative Planning Benchmark via Pseudo-Simulation},
  author    = {Cho, Hoonhee and Kang, Jae-Young and Lee, Giwon and Yang, Hyemin and Park, Heejun and Yoon, Kuk-jin},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026},
}
```

## Acknowledgments

Built on [NAVSIM](https://github.com/autonomousvision/navsim),
[nuPlan](https://github.com/motional/nuplan-devkit), and the
[V2X-Real](https://mobility-lab.seas.ucla.edu/v2x-real/) dataset. From KAIST,
Visual Intelligence Lab.
