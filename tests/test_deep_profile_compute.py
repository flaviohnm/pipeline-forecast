import pytest

from src.models.deep_profile import (
    DEFAULT_DEEP_CONFIG,
    build_models,
    build_nhits,
    load_deep_config,
    model_horizons,
    pick_model_horizon,
)
from src.utils.compute import describe_compute, get_trainer_kwargs, resolve_accelerator

GPU = {"available": True, "name": "NVIDIA Test GPU", "count": 1, "torch": "2.x", "cuda": "13.0", "reason": ""}
NO_GPU = {"available": False, "name": None, "count": 0, "torch": "2.x", "cuda": None,
          "reason": "PyTorch instalado sem suporte a CUDA (build somente CPU)"}
INFO = {"max_horizon": 720, "forecast_horizon": [24, 48, 168, 336, 720]}


# --- compute -----------------------------------------------------------------


def test_auto_uses_gpu_when_available_and_cpu_otherwise():
    assert resolve_accelerator("auto", GPU)[0] == "gpu"
    assert resolve_accelerator("auto", NO_GPU)[0] == "cpu"


def test_cpu_is_forced_even_with_gpu():
    acc, reason = resolve_accelerator("cpu", GPU)
    assert acc == "cpu" and "forçado" in reason


def test_gpu_requested_without_gpu_falls_back_to_cpu_with_warning():
    acc, reason = resolve_accelerator("gpu", NO_GPU)
    assert acc == "cpu" and "usando CPU" in reason
    assert describe_compute({"compute": {"accelerator": "gpu"}}, NO_GPU).startswith("⚠️")
    assert describe_compute({"compute": {"accelerator": "auto"}}, NO_GPU).startswith("🖥️")


def test_invalid_accelerator_is_rejected():
    with pytest.raises(ValueError, match="accelerator"):
        resolve_accelerator("tpu", GPU)


def test_trainer_kwargs():
    assert get_trainer_kwargs({"compute": {"accelerator": "auto", "devices": 2}}, GPU) == {"accelerator": "gpu", "devices": 2}
    assert get_trainer_kwargs({"compute": {"accelerator": "auto", "devices": 2}}, NO_GPU) == {"accelerator": "cpu", "devices": 1}
    assert get_trainer_kwargs(None, NO_GPU) == {"accelerator": "cpu", "devices": 1}


# --- deep_profile -----------------------------------------------------------


def test_defaults_when_section_missing():
    assert load_deep_config(None) == DEFAULT_DEEP_CONFIG
    assert load_deep_config({})["strategy"] == "per_horizon"


def test_override_merges_nested_values():
    cfg = load_deep_config({"deep_learning": {"input_size": 96, "models": {"Informer": {"n_head": 4}}}})
    assert cfg["input_size"] == 96
    assert cfg["models"]["Informer"]["n_head"] == 4
    assert cfg["models"]["Informer"]["hidden_size"] == 512  # preservado


@pytest.mark.parametrize(
    "bad",
    [{"strategy": "direct"}, {"common": {"loss": "huber"}}, {"input_size": 0}, {"models": {"Foo": {}}}],
)
def test_invalid_config_is_rejected(bad):
    with pytest.raises(ValueError):
        load_deep_config({"deep_learning": bad})


def test_model_horizons_by_strategy():
    assert model_horizons(INFO, load_deep_config({})) == [24, 48, 168, 336, 720]
    single = load_deep_config({"deep_learning": {"strategy": "single_truncated"}})
    assert model_horizons(INFO, single) == [720]


def test_pick_model_horizon():
    assert pick_model_horizon([24, 48, 720], 48) == 48
    assert pick_model_horizon([720], 24) == 720
    with pytest.raises(ValueError):
        pick_model_horizon([24], 48)


def test_build_models_apply_profile():
    cfg = {"random_seed": 1, "compute": {"accelerator": "cpu"}}
    models = build_models(48, cfg, input_size=96, extra_kwargs={"enable_progress_bar": False, "logger": False})
    assert list(models) == ["MLP", "LSTM", "NHITS", "Informer"]
    assert all(m.h == 48 and m.input_size == 96 for m in models.values())
    assert models["Informer"].hparams.hidden_size == 512
    assert models["Informer"].hparams.n_head == 8
    assert models["NHITS"].hparams.scaler_type == "standard"
    assert build_nhits(24, cfg, 96).h == 24
