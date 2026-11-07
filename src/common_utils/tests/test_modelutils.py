from yapit.common_utils.modelutils import load_model


def test_model_factory_supports_local_sdk_without_a_cache_keyword(tmp_path):
    class LocalModel:
        def __init__(self, path, *, device):
            self.path = path
            self.device = device

    path = tmp_path / "local-model.gguf"
    model = load_model(LocalModel, path, cache_parameter=None, device="cpu")
    assert model.path == str(path)
    assert model.device == "cpu"
