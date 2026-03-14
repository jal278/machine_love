"""Tests for LM provider (mocked API calls)."""
import pytest
from unittest.mock import MagicMock, patch


def test_openai_provider_complete(mocker):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "  Yes  "
    mock_client.chat.completions.create.return_value = mock_response

    mocker.patch(
        "maslow.lm.openai_provider.OpenAI",
        return_value=mock_client,
    )

    from maslow.lm.openai_provider import OpenAIProvider
    provider = OpenAIProvider(model="gpt-4o-mini", api_key="fake-key")
    result = provider.complete("Is this a test?")

    assert result == "Yes"
    mock_client.chat.completions.create.assert_called_once()


def test_openai_provider_chat(mocker):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Hello"
    mock_client.chat.completions.create.return_value = mock_response

    mocker.patch(
        "maslow.lm.openai_provider.OpenAI",
        return_value=mock_client,
    )

    from maslow.lm.openai_provider import OpenAIProvider
    provider = OpenAIProvider(model="gpt-4o-mini", api_key="fake-key")
    messages = [{"role": "user", "content": "Hi"}]
    result = provider.chat(messages, temperature=0.3, max_tokens=50)

    assert result == "Hello"
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == "gpt-4o-mini"
    assert call_kwargs["temperature"] == 0.3
    assert call_kwargs["max_tokens"] == 50


def test_openai_provider_uses_correct_model(mocker):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "ok"
    mock_client.chat.completions.create.return_value = mock_response

    mocker.patch(
        "maslow.lm.openai_provider.OpenAI",
        return_value=mock_client,
    )

    from maslow.lm.openai_provider import OpenAIProvider
    provider = OpenAIProvider(model="gpt-4o", api_key="fake-key")
    provider.complete("test")

    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == "gpt-4o"
