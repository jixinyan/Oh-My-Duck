import inspect
from pathlib import Path

from sam3.model_builder import build_sam3_predictor
import torch


def load_predictor(checkpoint: Path):
    predictor = build_sam3_predictor(checkpoint_path=str(checkpoint), version="sam3.1",
                                    compile=False, use_fa3=False, async_loading_frames=False)
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    weights = checkpoint_data.get("model", checkpoint_data)
    loaded = predictor.model.state_dict()
    missing = loaded.keys() - weights.keys()
    if len(missing) != 64 or weights.keys() - loaded.keys():
        raise ValueError("SAM 3.1 checkpoint keys differ from the pinned model")
    for name in missing:
        if not name.startswith("detector.backbone.vision_backbone.trunk.blocks."):
            raise ValueError(f"SAM checkpoint lacks model weight: {name}")
        stem, component = name.rsplit("_", 1)
        if not stem.endswith(".attn.freqs_cis") or component not in {"real", "imag"}:
            raise ValueError(f"Unexpected derived SAM buffer: {name}")
        source = weights[stem]
        expected_buffer = source.real if component == "real" else source.imag
        if not torch.is_complex(source) or not torch.equal(loaded[name].cpu(), expected_buffer):
            raise ValueError(f"SAM RoPE buffer differs from the checkpoint: {name}")
    for name, value in weights.items():
        if loaded[name].shape != value.shape or loaded[name].dtype != value.dtype:
            raise ValueError(f"SAM checkpoint tensor metadata differs: {name}")
    original = predictor.model.init_state
    expected = ("resource_path", "offload_video_to_cpu", "async_loading_frames",
                "use_torchcodec", "use_cv2", "input_is_mp4")
    if tuple(inspect.signature(original).parameters) != expected:
        raise ValueError("Pinned SAM 3.1 multiplex init_state signature differs")

    # 固定版本的公共 session 接口包含 multiplex 未接受的 state 参数。
    def init_state(resource_path, offload_video_to_cpu=False, offload_state_to_cpu=False,
                   async_loading_frames=False):
        if offload_state_to_cpu:
            raise ValueError("SAM 3.1 multiplex does not support state offloading")
        return original(resource_path=resource_path, offload_video_to_cpu=offload_video_to_cpu,
                        async_loading_frames=async_loading_frames)

    predictor.model.init_state = init_state
    return predictor
