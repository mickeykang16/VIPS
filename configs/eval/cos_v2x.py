# Eval config: CoS-V2X cooperative model (built on SparseDrive).
#
# All cos_v2x settings live here. The model is expected at models/CoS-V2X (clone
# or symlink it there per the README); only the machine-specific V2X-Real *data*
# paths live in configs/eval/paths.py. Anything set explicitly in the environment
# overrides the values here.
#
# Usage:
#   CONFIG=configs/eval/cos_v2x.py \
#     bash scripts/evaluation/run_cv_pdm_score_evaluation_v2xreal_stage2_coop_temporal_parallel.sh

import os

AGENT = "cos_v2x"
AGENT_CLASS_PATH = "navsim.agents.cos_v2x_agent.CoSV2XAgent"
CONDA_ENV = "sparsedrive_navsimv2"  # the CoS-V2X model conda env (see CoS-V2X README)
COS_V2X_MODE = "coop"               # "coop" (vehicle + infrastructure) or "veh"

# Model checkout + checkpoint (clone/symlink into models/CoS-V2X per the README).
COS_V2X_FOLDER = "models/CoS-V2X"
COS_V2X_CONFIG_PATH = os.path.join(
    COS_V2X_FOLDER, "projects/configs/sparsedrive_small_stage2_6cams_v2x_top100.py"
)
COS_V2X_MODEL_CHECKPOINT_PATH = os.path.join(
    COS_V2X_FOLDER, "work_dirs/6cams_both_infra_v8_v2x_stage2_top100_fix/latest.pth"
)

TRAFFIC_POLICY = "log_replay"  # reported numbers; idm / constant_velocity optional
GPU_IDS = "0,1,2,3"
NUM_WORKERS = 4
VISUALIZE = 0
