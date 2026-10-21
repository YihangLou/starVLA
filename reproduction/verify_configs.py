"""Static verification only; never labels a model's runtime test as passed."""
import ast
import json
from pathlib import Path
import subprocess

import yaml

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
matrix = json.loads((ROOT / "run_matrix.json").read_text())
registered = {}
for path in (SOURCE / "starVLA/model/framework/VLM4A").glob("*.py"):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for dec in node.decorator_list:
                if (
                    isinstance(dec, ast.Call)
                    and isinstance(dec.func, ast.Attribute)
                    and dec.func.attr == "register"
                    and dec.args
                    and isinstance(dec.args[0], ast.Constant)
                ):
                    registered[dec.args[0].value] = str(path.relative_to(SOURCE))

registry = ast.parse((SOURCE / "examples/LIBERO/train_files/data_registry/data_config.py").read_text())
mixtures = next(
    ast.literal_eval(n.value) for n in registry.body
    if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "DATASET_NAMED_MIXTURES" for t in n.targets)
)
assert len(mixtures["libero_all"]) == 4
checks = []
for item in matrix:
    cfg = yaml.safe_load(Path(item["config"]).read_text())
    framework = cfg["framework"]
    assert framework["name"] in registered
    assert framework["qwenvl"]["attn_implementation"] == "flash_attention_2"
    assert framework["action_model"]["action_dim"] == 7
    assert framework["action_model"]["action_horizon"] == 8
    assert framework["action_model"]["future_action_window_size"] == 7
    assert cfg["datasets"]["vla_data"]["data_mix"] == "libero_all"
    assert cfg["datasets"]["vla_data"]["per_device_batch_size"] == 16
    vlm = cfg["datasets"]["vlm_data"]
    assert vlm["model_type"] == "qwen3vl"
    assert 0 < vlm["video_min_frames"] <= vlm["video_max_frames"]
    assert 0 < vlm["video_min_pixels"] <= vlm["video_max_pixels"]
    assert vlm["video_fps"] > 0
    assert cfg["trainer"]["max_train_steps"] == 100000
    assert cfg["trainer"]["save_interval"] == 10000
    assert cfg["trainer"]["learning_rate"] == {
        "base": 2.5e-5, "qwen_vl_interface": 1e-5, "action_model": 1e-4,
    }
    trainer = SOURCE / "starVLA/training" / item["trainer"]
    ast.parse(trainer.read_text())
    cotrain = item["run"].endswith("_cotrain")
    assert ("vlm_train_dataloader = build_dataloader" in trainer.read_text()) == cotrain
    checks.append({
        "run": item["run"], "static_config": "pass",
        "implementation": registered[framework["name"]],
        "runtime": "not_tested",
    })
for name in ("launch_train.sh", "start_policy_server.sh", "evaluate_all.sh"):
    subprocess.run(["bash", "-n", str(ROOT / name)], check=True)
report = {
    "static_checks": checks,
    "libero_all_suites": [row[0] for row in mixtures["libero_all"]],
    "shell_syntax": "pass",
    "warning": "Static checks do not establish model loading, forward/backward, training, or evaluation success.",
}
(ROOT / "config_verification.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
