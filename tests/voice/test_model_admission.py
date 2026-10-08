import os
from pathlib import Path
import sys
import tempfile

import pytest
import torch

from oh_my_duck.voice.qwen import (
    QwenSpeechRecognition, QwenSpeechSynthesis, QwenVoiceDesign, _model_options,
)


def test_cpu_options_preserve_float32_without_cuda():
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert not torch.cuda.is_initialized()
    assert _model_options("cpu") == {"device_map": "cpu", "dtype": torch.float32}
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("device", (None, True, 1, "", " ", "cpu:0", "mps", "meta",
                                    "invalid-device", "cuda:abc", "cuda:-1", "cudabad"))
def test_actual_constructors_reject_device_before_model_import(device):
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert not torch.cuda.is_initialized()
    temporary = Path.cwd() / ".cache/tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temporary) as directory:
        output = Path(directory) / "audio"
        for constructor in (QwenSpeechRecognition, QwenSpeechSynthesis, QwenVoiceDesign):
            with pytest.raises((ValueError, RuntimeError)):
                if constructor is QwenSpeechRecognition:
                    constructor(device=device)
                else:
                    constructor(output, device=device)
            assert not output.exists()
            assert not torch.cuda.is_initialized()
            assert "qwen_asr" not in sys.modules and "qwen_tts" not in sys.modules
