# Eval config: ADMLP planner (custom agent class via AGENT_CLASS_PATH).
#
# All ADMLP settings live here; only the machine-specific V2X-Real *data* paths
# live in configs/eval/paths.py. Explicit environment values override these.

AGENT = "human"  # base agent name; the real planner is the custom class below
AGENT_CLASS_PATH = "navsim.agents.admlp_sim_v2.ADMLPSim"
CONDA_ENV = "sparsedrive_navsimv2"  # activate this env before running

# Trained ADMLP weights (place them under checkpoints/admlp/, or edit these paths).
ADMLP_V2_CKPT_PATH = "checkpoints/admlp/admlp_univ2x.pt"
ADMLP_V2_STATS_PATH = "checkpoints/admlp/admlp_univ2x_train_stats.pkl"

TRAFFIC_POLICY = "log_replay"  # reported numbers; idm / constant_velocity optional
GPU_IDS = "0,1,2,3"
NUM_WORKERS = 4
VISUALIZE = 0
