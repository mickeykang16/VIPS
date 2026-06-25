# Eval config: constant-velocity baseline (no model, no checkpoint).

AGENT = "constant_velocity"
CONDA_ENV = "vips"

TRAFFIC_POLICY = "log_replay"
GPU_IDS = "0,1,2,3"
NUM_WORKERS = 8
VISUALIZE = 0
