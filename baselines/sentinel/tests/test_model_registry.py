import json

import sentinelrag.utils.model_registry as model_registry
from sentinelrag.utils.model_registry import load_model_preset


def test_load_model_preset_uses_llm_schema(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    preset_path = models_dir / "example.json"
    preset_path.write_text(
        json.dumps(
            {
                "llm_url": "https://example.test/v1",
                "api_key": "test-key",
                "model": "example-model",
                "llm_arg": {"temperature": 0},
            }
        ),
        encoding="utf-8",
    )

    preset = load_model_preset("example", models_dir=models_dir)

    assert preset.preset_name == "example"
    assert preset.llm_url == "https://example.test/v1"
    assert preset.api_key == "test-key"
    assert preset.model == "example-model"
    assert preset.llm_arg == {"temperature": 0}


def test_create_llm_client_resolves_api_key_reference(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    preset_path = models_dir / "example.json"
    preset_path.write_text(
        json.dumps(
            {
                "llm_url": "https://example.test/v1",
                "api_key": "${SENTINELRAG_TEST_API_KEY}",
                "model": "example-model",
                "llm_arg": {"temperature": 0},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SENTINELRAG_TEST_API_KEY", "resolved-key")

    class FakeOpenAI:
        def __init__(self, api_key, base_url):
            self.api_key = api_key
            self.base_url = base_url

    monkeypatch.setattr(model_registry.openai, "OpenAI", FakeOpenAI, raising=False)

    llm_client, preset = model_registry.create_llm_client_from_preset(
        "example", models_dir=models_dir, include_async=False
    )

    assert preset.model == "example-model"
    assert llm_client.model == "example-model"
    assert llm_client.llm_arg == {"temperature": 0}
    assert llm_client.client.api_key == "resolved-key"
    assert llm_client.client.base_url == "https://example.test/v1"
