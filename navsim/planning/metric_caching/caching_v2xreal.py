import gc
import io
import logging
import os
import uuid
import pickle
import contextlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from hydra.utils import instantiate
from nuplan.planning.training.experiments.cache_metadata_entry import (
    CacheMetadataEntry,
    CacheResult,
    save_cache_metadata,
)
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from nuplan.planning.utils.multithreading.worker_utils import worker_map
from omegaconf import DictConfig

from nuplan.common.actor_state.vehicle_parameters import VehicleParameters

from navsim.common.dataclasses import Scene, SensorConfig
from navsim.common.dataloader_v2xreal import SceneLoaderV2XReal, load_test_tokens_from_pkl
from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario


def _get_v2xreal_ego_parameters() -> VehicleParameters:
    """V2X-Real ego: ego2global_translation = vehicle center → rear_axle_to_center = 0."""
    half_len = 4.5 / 2.0  # EGO_LENGTH / 2
    return VehicleParameters(
        vehicle_name="v2xreal_ego",
        vehicle_type="v2xreal",
        width=2.0,
        front_length=half_len,
        rear_length=half_len,
        wheel_base=2.0,
        cog_position_from_rear_axle=1.0,
        height=1.8,
    )

logger = logging.getLogger(__name__)


def _load_scene_tokens_from_file(path: Path) -> List[str]:
    """Load scene tokens from a text file (one token per line)."""
    if not path.exists():
        raise FileNotFoundError(f"SCENE_TOKENS_FILE not found: {path}")

    tokens: List[str] = []
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            tokens.append(line)

    if not tokens:
        raise ValueError(f"SCENE_TOKENS_FILE is empty after parsing: {path}")
    return tokens


def _load_frame_tokens_by_scene_from_pkl(pkl_path: Path, target_scene_tokens: Set[str]) -> List[str]:
    """Expand scene tokens to frame tokens from V2X-Real PKL infos."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    infos = data.get("infos", [])
    frame_tokens: List[str] = []
    matched_scenes: Set[str] = set()
    for info in infos:
        scene_token = info.get("scene_token")
        token = info.get("token")
        if scene_token in target_scene_tokens and token is not None:
            frame_tokens.append(token)
            matched_scenes.add(scene_token)

    missing = sorted(target_scene_tokens - matched_scenes)
    logger.info(f"Scene-token filter: requested={len(target_scene_tokens)}, matched={len(matched_scenes)}")
    if missing:
        logger.warning(f"Scene-token filter: unmatched scenes ({len(missing)}): {missing}")
    logger.info(f"Scene-token filter: expanded to frame tokens={len(frame_tokens)}")

    if not frame_tokens:
        raise ValueError(
            "Scene-token filter produced 0 frame tokens. "
            "Check SCENE_TOKENS_FILE against pkl infos.scene_token."
        )
    return frame_tokens


def cache_scenarios(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[CacheResult]:
    """
    Performs the caching of scenario DB files in parallel for V2X-Real dataset.
    :param args: A list of dicts containing the following items:
        "cfg": the DictConfig to use to process the file.
        "log_file": log/scene name
        "tokens": list of tokens to process
    :return: A list of CacheResult with statistics
    """

    def cache_scenarios_internal(args: List[Dict[str, Union[Path, DictConfig]]]) -> List[CacheResult]:
        def cache_single_scenario(
            scene: Scene, processor: MetricCacheProcessor
        ) -> Optional[CacheMetadataEntry]:
            scenario = NavSimScenario(
                scene,
                map_root=os.environ["NUPLAN_MAPS_ROOT"],
                map_version="nuplan-maps-v1.0",
                ego_vehicle_parameters=_get_v2xreal_ego_parameters(),
            )

            return processor.compute_and_save_metric_cache(scenario)

        node_id = int(os.environ.get("NODE_RANK", 0))
        thread_id = str(uuid.uuid4())

        log_names = [a["log_file"] for a in args]
        # Flatten per-log token batches passed from cache_data() and deduplicate.
        tokens = list(dict.fromkeys([t for a in args for t in a["tokens"]]))
        cfg: DictConfig = args[0]["cfg"]

        # Load V2X-Real scenes from pkl
        # Try to get map root from environment or use cfg
        map_root = os.getenv('NUPLAN_MAPS_ROOT') or Path(cfg.v2xreal_data_root)
        
        # Suppress verbose per-worker dataloader prints/tqdm to keep Ray logs readable.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            scene_loader = SceneLoaderV2XReal(
                pkl_path=Path(cfg.v2xreal_pkl_path),
                scene_filter=instantiate(cfg.train_test_split.scene_filter),
                sensor_config=SensorConfig.build_no_sensors(),
                sensor_blob_path=None,  # No sensors needed for caching
                map_root=Path(map_root),  # Pass map root for map loading
                connector_cache_dir = cfg.metric_cache_path,
                connector_force_recompute = False,
            )
        available_tokens = set(scene_loader.tokens)
        target_tokens = [t for t in tokens if t in available_tokens]
        missing_tokens = len(tokens) - len(target_tokens)
        if missing_tokens > 0:
            logger.warning(
                f"Worker token mismatch: requested={len(tokens)}, matched={len(target_tokens)}, missing={missing_tokens} "
                f"(thread_id={thread_id}, node_id={node_id})"
            )

        # Create metric cache processor
        assert cfg.metric_cache_path is not None, f"Cache path cannot be None when caching, got {cfg.metric_cache_path}"
        
        use_pdms_v1 = os.getenv("PDMS_V2", "true").lower() != "true"

        processor = MetricCacheProcessor(
            cache_path=cfg.metric_cache_path,
            force_feature_computation=cfg.force_feature_computation,
            proposal_sampling=instantiate(cfg.proposal_sampling),
            use_pdms_v1=use_pdms_v1,
        )

        logger.info(
            f"Extracted {len(scene_loader)} scenarios for thread_id={thread_id}, node_id={node_id} "
            f"(requested tokens={len(tokens)}, matched tokens={len(target_tokens)}, logs={len(log_names)})."
        )
        logger.info(
            f"Worker start: thread_id={thread_id}, node_id={node_id}, "
            f"targets={len(target_tokens)}"
        )
        num_failures = 0
        num_successes = 0
        all_file_cache_metadata: List[Optional[CacheMetadataEntry]] = []
        
        # For V2X-Real with NuScenes loader, iterate through tokens
        if False:
            # target = "2023-04-04-14-27-53_44_0_folder_1_-2"
            target = '2023-03-17-16-11-12_2_0_folder_1_-2'
            # 2023-04-04-13-58-53_15_0_folder_2_-1
            # target = "2023-04-05-16-31-26_28_1"
            filtered_tokens = [t for t in target_tokens if t.startswith(target)]
            logger.info(f"DEBUG_MODE=hm: filtered tokens {len(filtered_tokens)} / {len(target_tokens)}")

            for idx, token in enumerate(filtered_tokens, start=1):
                logger.info(
                    f"Processing scenario {idx} / {len(filtered_tokens)} (token={token}) in thread_id={thread_id}, node_id={node_id}"
                )
                # Get scene from token
                scene = scene_loader.get_scene_from_token(token)
                file_cache_metadata = cache_single_scenario(scene, processor)
                gc.collect()

                num_failures += 0 if file_cache_metadata else 1
                num_successes += 1 if file_cache_metadata else 0
                all_file_cache_metadata += [file_cache_metadata]

            logger.info(f"Finished processing scenarios for thread_id={thread_id}, node_id={node_id}")
            return [
                CacheResult(
                    failures=num_failures,
                    successes=num_successes,
                    cache_metadata=all_file_cache_metadata,
                )
            ]

        total_targets = len(target_tokens)
        progress_every = max(1, total_targets // 10) if total_targets > 0 else 1
        for idx, token in enumerate(target_tokens, start=1):
            # Get scene from token
            scene = scene_loader.get_scene_from_token(token)
            file_cache_metadata = cache_single_scenario(scene, processor)
            gc.collect()

            num_failures += 0 if file_cache_metadata else 1
            num_successes += 1 if file_cache_metadata else 0
            all_file_cache_metadata += [file_cache_metadata]

            if idx == 1 or idx == total_targets or idx % progress_every == 0:
                logger.info(
                    f"Worker progress: thread_id={thread_id}, node_id={node_id}, "
                    f"{idx}/{total_targets} ({100.0 * idx / max(total_targets, 1):.0f}%)"
                )

        logger.info(f"Finished processing scenarios for thread_id={thread_id}, node_id={node_id}")
        return [
            CacheResult(
                failures=num_failures,
                successes=num_successes,
                cache_metadata=all_file_cache_metadata,
            )
        ]

    result = cache_scenarios_internal(args)
    gc.collect()
    return result


def cache_data(cfg: DictConfig, worker: WorkerPool) -> None:
    """
    Build the V2X-Real datamodule and cache all samples.
    :param cfg: omegaconf dictionary
    :param worker: Worker to submit tasks which can be executed in parallel
    """
    assert cfg.metric_cache_path is not None, f"Cache path cannot be None when caching, got {cfg.metric_cache_path}"
    assert cfg.v2xreal_pkl_path is not None, f"V2X-Real pkl path must be specified, got {cfg.get('v2xreal_pkl_path')}"
    

    # Load frame tokens to filter by.
    scene_tokens_file = os.getenv("SCENE_TOKENS_FILE")
    if scene_tokens_file:
        scene_tokens_path = Path(scene_tokens_file)
        scene_tokens = _load_scene_tokens_from_file(scene_tokens_path)
        logger.info(f"Using scene-token filter file: {scene_tokens_path} ({len(scene_tokens)} scenes)")
        test_tokens = _load_frame_tokens_by_scene_from_pkl(Path(cfg.v2xreal_pkl_path), set(scene_tokens))
    else:
        # Default behavior: all test tokens from pkl.
        test_tokens = load_test_tokens_from_pkl(Path(cfg.v2xreal_pkl_path))

    # Load V2X-Real scenes from pkl
    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    
    # Filter to only test tokens
    scene_filter.tokens = test_tokens
    
    # Try to get map root from environment or use cfg
    map_root = os.getenv('NUPLAN_MAPS_ROOT') or Path(cfg.v2xreal_data_root)
    
    scene_loader = SceneLoaderV2XReal(
        pkl_path=Path(cfg.v2xreal_pkl_path),
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
        sensor_blob_path=None,
        map_root=Path(map_root),  # Pass map root for map loading
        connector_cache_dir = cfg.metric_cache_path,
        connector_force_recompute = True,
    )

    scene_loader.precompute_from_scene(
    visualize_every=1,
    visualize_out_dir=Path(cfg.metric_cache_path) / "_lane_connector_vis",
    )

    data_points = [
        {
            "cfg": cfg,
            "log_file": log_file,
            "tokens": tokens_list,
        }
        for log_file, tokens_list in scene_loader.get_tokens_list_per_log().items()
    ]
    total_requested_tokens = sum(len(dp["tokens"]) for dp in data_points)
    logger.info(
        "Stage1 dispatch summary: logs=%s, scenario_tokens=%s",
        str(len(data_points)),
        str(total_requested_tokens),
    )
    logger.info("Starting metric caching of %s scenes...", str(len(scene_loader)))

    cache_results = worker_map(worker, cache_scenarios, data_points)

    num_success = sum(result.successes for result in cache_results)
    num_fail = sum(result.failures for result in cache_results)
    num_total = num_success + num_fail
    if num_fail == 0:
        logger.info(
            "Completed dataset caching! All %s features and targets were cached successfully.",
            str(num_total),
        )
    else:
        logger.info(
            "Completed dataset caching! Failed features and targets: %s out of %s",
            str(num_fail),
            str(num_total),
        )

    cached_metadata = [
        cache_metadata_entry
        for cache_result in cache_results
        for cache_metadata_entry in cache_result.cache_metadata
        if cache_metadata_entry is not None
    ]

    node_id = int(os.environ.get("NODE_RANK", 0))
    logger.info(f"Node {node_id}: Storing metadata csv file containing cache paths for valid features and targets...")
    save_cache_metadata(cached_metadata, Path(cfg.metric_cache_path), node_id)
    logger.info("Done storing metadata csv file.")
