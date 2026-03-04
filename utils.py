import os
import json
from model_deploy import ConvNextLit_Inference
import torch

def load_json_mapping(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    print(f"Warning: Mapping file not found at {filepath}")
    return {}

def load_inference_model(pt_path, num_classes, dropout=0.3):
    model = ConvNextLit_Inference(num_classes=num_classes, dropout=dropout)
    state_dict = torch.load(pt_path, map_location="cpu")

    if not isinstance(state_dict, dict) or len(state_dict) == 0:
        raise RuntimeError(f"Invalid .pt at {pt_path}: expected a non-empty state_dict dict.")
    
    if isinstance(state_dict.get("state_dict"), dict):
        raise RuntimeError(
            f"Got a Lightning checkpoint at {pt_path}. "
            "Please use the converted .pt exported from model.state_dict() or base.state_dict()."
        )

    try:
        model.model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        sample_keys = list(state_dict.keys())[:10]
        raise RuntimeError(
            f"Strict load failed for {pt_path}. "
            f"Expected keys for inner ConvNeXt (e.g., 'features.*', 'classifier.*'). "
            f"Sample keys: {sample_keys}"
        ) from exc

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model
