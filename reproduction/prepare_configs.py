"""Produce explicit configs for the shared StarVLA repository."""
import copy
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
base = yaml.safe_load((SOURCE / "examples/LIBERO/train_files/starvla_cotrain_libero.yaml").read_text())
base["run_root_dir"] = str(ROOT / "results/Checkpoints")
base["wandb_entity"] = None
base["wandb_project"] = "starvla-guideline-reproduction"
base["datasets"]["vla_data"]["data_root_dir"] = str(SOURCE / "playground/Datasets/LEROBOT_LIBERO_DATA")
# The example YAML predates Qwen3-VL and omits video processor settings that
# this loader reads even for image-only COCO data. Keep the installed Qwen3
# processor defaults (transformers 4.57.0 / downloaded video config).
base["datasets"]["vlm_data"].update(
    model_type="qwen3vl", video_min_frames=4, video_max_frames=768,
    video_fps=2, video_min_pixels=4096, video_max_pixels=25165824,
)
base["trainer"]["save_interval"] = 10000
base["trainer"]["freeze_modules"] = ""
matrix = []
for framework in ("QwenOFT", "QwenFast", "QwenPI", "QwenGR00T"):
    for cotrain in (False, True):
        cfg = copy.deepcopy(base)
        cfg["framework"]["name"] = framework
        model = "Qwen3-VL-4B-Instruct-Action" if framework == "QwenFast" else "Qwen3-VL-4B-Instruct"
        cfg["framework"]["qwenvl"]["base_vlm"] = str(SOURCE / "playground/Pretrained_models" / model)
        cfg["framework"]["action_model"]["future_action_window_size"] = 7
        cfg["framework"]["action_model"]["past_action_window_size"] = 0
        run = framework + ("_cotrain" if cotrain else "_vla")
        cfg["run_id"] = run
        path = ROOT / "configs" / f"{run}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        matrix.append({
            "run": run, "framework": framework, "config": str(path),
            "trainer": "train_starvla_cotrain.py" if cotrain else "train_starvla.py",
            "runtime_status": "not_run",
        })
(ROOT / "run_matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
