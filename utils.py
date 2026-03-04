import os
import json
from model_deploy import ConvNextLit_Inference
import torch

def load_json_mapping(filepath):
    """โหลดไฟล์ JSON สำหรับแปลง Index เป็นชื่อคลาส"""
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    print(f"Warning: Mapping file not found at {filepath}")
    return {}

def _strip_prefix_once(state_dict, prefix):
    return {
        (k[len(prefix):] if isinstance(k, str) and k.startswith(prefix) else k): v
        for k, v in state_dict.items()
    }


def _add_prefix(state_dict, prefix):
    return {
        (k if isinstance(k, str) and k.startswith(prefix) else f"{prefix}{k}"): v
        for k, v in state_dict.items()
    }


def _filter_prefix(state_dict, prefix, strip=False):
    out = {}
    for k, v in state_dict.items():
        if isinstance(k, str) and k.startswith(prefix):
            out[k[len(prefix):] if strip else k] = v
    return out


def load_inference_model(pt_path, num_classes, dropout=0.3):
    model = ConvNextLit_Inference(num_classes=num_classes, dropout=dropout)

    raw_obj = torch.load(pt_path, map_location="cpu")
    # Support both plain state_dict (*.pt) and Lightning checkpoint dicts.
    state_dict = (
        raw_obj["state_dict"]
        if isinstance(raw_obj, dict) and isinstance(raw_obj.get("state_dict"), dict)
        else raw_obj
    )

    if not isinstance(state_dict, dict) or len(state_dict) == 0:
        raise RuntimeError(
            f"Invalid checkpoint format at {pt_path}: expected a non-empty state_dict."
        )

    attempts = [
        ("outer/raw", model, state_dict),
        ("inner/raw", model.model, state_dict),
        ("outer/strip_model_prefix", model, _strip_prefix_once(state_dict, "model.")),
        ("inner/strip_model_prefix", model.model, _strip_prefix_once(state_dict, "model.")),
        ("outer/add_model_prefix", model, _add_prefix(state_dict, "model.")),
        ("outer/filter_model_prefix", model, _filter_prefix(state_dict, "model.", strip=False)),
        (
            "inner/filter_model_prefix_strip",
            model.model,
            _filter_prefix(state_dict, "model.", strip=True),
        ),
        (
            "outer/filter_model_prefix_strip",
            model,
            _filter_prefix(state_dict, "model.", strip=True),
        ),
    ]

    errors = []
    loaded = False
    for name, target, candidate in attempts:
        if not isinstance(candidate, dict) or len(candidate) == 0:
            errors.append(f"{name}: empty candidate dict")
            continue
        try:
            # strict=True is required: every target parameter must come from the checkpoint.
            target.load_state_dict(candidate, strict=True)
            loaded = True
            break
        except RuntimeError as exc:
            errors.append(f"{name}: {exc}")

    if not loaded:
        preview = list(state_dict.keys())[:10]
        details = "\n".join(errors)
        raise RuntimeError(
            "Failed to load checkpoint with strict=True using known key mappings.\n"
            f"Path: {pt_path}\n"
            f"Sample keys: {preview}\n"
            f"Attempts:\n{details}"
        )

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model
