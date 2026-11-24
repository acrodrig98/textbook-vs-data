"""Environment checks that run on any machine (no GPU, no model downloads)."""
import shutil

import torch


def test_core_imports():
    import circuit_tracer  # noqa: F401
    from circuit_tracer import Graph, ReplacementModel, attribute  # noqa: F401
    import nnsight  # noqa: F401
    import transformer_lens  # noqa: F401


def test_pinned_versions():
    import huggingface_hub
    import transformers

    assert transformers.__version__ == "4.57.3"
    assert huggingface_hub.__version__ == "0.36.2"


def test_cli_tools_on_path():
    assert shutil.which("circuit-tracer"), "circuit-tracer CLI missing"
    assert shutil.which("hf"), "Hugging Face CLI missing"


def test_torch_basic_math():
    x = torch.randn(4, 4)
    assert torch.allclose(x @ torch.eye(4), x)
