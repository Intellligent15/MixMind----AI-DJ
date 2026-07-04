import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.llm.gemini import GeminiProvider


@pytest.mark.asyncio
async def test_gemini_provider_caching():
    # Mock storage
    mock_storage = AsyncMock()
    # Let exists() return True, and read() return a dummy plan
    dummy_plan = [{"tool": "set_transition_window", "from_song_time_start": 1.0, "to_song_time_start": 2.0, "duration_bars": 8}]
    mock_storage.exists.return_value = True
    mock_storage.read.return_value = json.dumps({"response": dummy_plan}).encode("utf-8")
    
    # Mock google-genai client
    mock_client = MagicMock()
    mock_generate = MagicMock()
    mock_client.models.generate_content = mock_generate

    with patch("app.services.llm.base.get_storage", return_value=mock_storage):
        with patch("app.services.llm.gemini.genai.Client", return_value=mock_client):
            provider = GeminiProvider(api_key="fake")
            plan = await provider.plan_transition(
                from_song={"bpm": 120},
                to_song={"bpm": 125},
                tools_schema="Tools desc"
            )
            
            assert plan == dummy_plan
            # Ensure generate_content was NOT called
            mock_generate.assert_not_called()
            mock_storage.exists.assert_called_once()
            mock_storage.read.assert_called_once()


@pytest.mark.asyncio
async def test_gemini_provider_invalid_json_raises():
    """The worker's fallback path relies on this exception bubbling up."""
    mock_storage = AsyncMock()
    mock_storage.exists.return_value = False

    mock_response = MagicMock()
    mock_response.text = "not valid json {{{"
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("app.services.llm.base.get_storage", return_value=mock_storage):
        with patch("app.services.llm.gemini.genai.Client", return_value=mock_client):
            provider = GeminiProvider(api_key="fake")
            with pytest.raises(ValueError, match="Invalid JSON"):
                await provider.plan_transition({}, {}, "Tools desc")
            # Did NOT cache garbage.
            mock_storage.write.assert_not_called()


@pytest.mark.asyncio
async def test_gemini_provider_non_list_raises():
    mock_storage = AsyncMock()
    mock_storage.exists.return_value = False

    mock_response = MagicMock()
    mock_response.text = json.dumps({"oops": "object not list"})
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("app.services.llm.base.get_storage", return_value=mock_storage):
        with patch("app.services.llm.gemini.genai.Client", return_value=mock_client):
            provider = GeminiProvider(api_key="fake")
            with pytest.raises(ValueError, match="Expected JSON object with"):
                await provider.plan_transition({}, {}, "Tools desc")
            # Valid-but-wrong-shape JSON IS cached (it is the model's real
            # answer; planner-level repair can still salvage it from cache).
            # Only undecodable JSON is kept out of the cache.
            mock_storage.write.assert_called_once()


@pytest.mark.asyncio
async def test_gemini_provider_generation():
    # Mock storage to simulate cache miss
    mock_storage = AsyncMock()
    mock_storage.exists.return_value = False
    
    dummy_plan = [{"tool": "set_transition_window", "from_song_time_start": 1.0, "to_song_time_start": 2.0, "duration_bars": 8}]
    
    # Mock google-genai client
    mock_response = MagicMock()
    mock_response.text = json.dumps(dummy_plan)
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("app.services.llm.base.get_storage", return_value=mock_storage):
        with patch("app.services.llm.gemini.genai.Client", return_value=mock_client):
            provider = GeminiProvider(api_key="fake")
            plan = await provider.plan_transition(
                from_song={"bpm": 120},
                to_song={"bpm": 125},
                tools_schema="Tools desc"
            )
            
            assert plan == dummy_plan
            mock_client.models.generate_content.assert_called_once()
            
            # Check if prompt + response were saved
            mock_storage.write.assert_called_once()
            written_key = mock_storage.write.call_args[0][0]
            assert written_key.startswith("mix_plan_logs/")
            written_bytes = mock_storage.write.call_args[0][1]
            written_data = json.loads(written_bytes.decode("utf-8"))
            # The shared base caches the RAW model response (the {"plan": ...}
            # wrapper), not the unwrapped list — complete_json() is generic
            # and the unwrap happens a layer above.
            assert written_data["response"] == dummy_plan
