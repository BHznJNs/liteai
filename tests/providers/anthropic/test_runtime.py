import pytest

from dais_sdk.providers.anthropic import AnthropicProvider


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("https://example.com", "https://example.com"),
        ("https://example.com/", "https://example.com"),
        ("https://example.com/v1", "https://example.com"),
        ("https://example.com/v1/", "https://example.com"),
        ("https://example.com/custom", "https://example.com/custom"),
        ("https://example.com/custom/", "https://example.com/custom"),
        ("https://v1.com/", "https://v1.com"),
        ("https://example.v1/", "https://example.v1"),
    ],
)
def test_anthropic_provider_base_url_preprocess(base_url: str, expected: str) -> None:
    assert AnthropicProvider._base_url_preprocess(base_url) == expected
