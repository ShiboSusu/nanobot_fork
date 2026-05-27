"""Create LLM providers from config."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nanobot.config.schema import Config, ProviderConfig
from nanobot.providers.base import GenerationSettings, LLMProvider
from nanobot.providers.registry import ProviderSpec, find_by_name


@dataclass(frozen=True)
class ProviderSnapshot:
    provider: LLMProvider
    model: str
    context_window_tokens: int
    signature: tuple[object, ...]


def _resolve_provider(
    config: Config,
    model: str,
    provider_override: str | None = None,
) -> tuple[ProviderConfig | None, str | None, ProviderSpec | None]:
    """Resolve provider config/name/spec for a model, optionally forcing the provider."""
    if provider_override and provider_override != "auto":
        spec = find_by_name(provider_override)
        if spec is None:
            raise ValueError(f"Unknown provider '{provider_override}'.")
        p = getattr(config.providers, spec.name, None)
        return p, spec.name, spec

    provider_name = config.get_provider_name(model)
    p = config.get_provider(model)
    spec = find_by_name(provider_name) if provider_name else None
    return p, provider_name, spec


def _api_base_for_provider(
    config: Config,
    model: str,
    p: ProviderConfig | None,
    spec: ProviderSpec | None,
    *,
    provider_override: str | None = None,
) -> str | None:
    if p and p.api_base:
        return p.api_base
    if provider_override and provider_override != "auto":
        return spec.default_api_base if spec and spec.default_api_base else None
    return config.get_api_base(model)


def make_provider(
    config: Config,
    *,
    model_override: str | None = None,
    provider_override: str | None = None,
) -> LLMProvider:
    """Create the LLM provider implied by config, with optional model/provider override."""
    model = model_override or config.agents.defaults.model
    p, provider_name, spec = _resolve_provider(config, model, provider_override)
    backend = spec.backend if spec else "openai_compat"
    api_base = _api_base_for_provider(config, model, p, spec, provider_override=provider_override)

    if backend == "azure_openai":
        if not p or not p.api_key or not p.api_base:
            raise ValueError("Azure OpenAI requires api_key and api_base in config.")
    elif backend == "openai_compat" and not model.startswith("bedrock/"):
        needs_key = not (p and p.api_key)
        exempt = spec and (spec.is_oauth or spec.is_local or spec.is_direct)
        if needs_key and not exempt:
            raise ValueError(f"No API key configured for provider '{provider_name}'.")

    if backend == "openai_codex":
        from nanobot.providers.openai_codex_provider import OpenAICodexProvider

        provider = OpenAICodexProvider(default_model=model)
    elif backend == "azure_openai":
        from nanobot.providers.azure_openai_provider import AzureOpenAIProvider

        provider = AzureOpenAIProvider(
            api_key=p.api_key,
            api_base=p.api_base,
            default_model=model,
        )
    elif backend == "github_copilot":
        from nanobot.providers.github_copilot_provider import GitHubCopilotProvider

        provider = GitHubCopilotProvider(default_model=model)
    elif backend == "anthropic":
        from nanobot.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider(
            api_key=p.api_key if p else None,
            api_base=api_base,
            default_model=model,
            extra_headers=p.extra_headers if p else None,
        )
    else:
        from nanobot.providers.openai_compat_provider import OpenAICompatProvider

        provider = OpenAICompatProvider(
            api_key=p.api_key if p else None,
            api_base=api_base,
            default_model=model,
            extra_headers=p.extra_headers if p else None,
            spec=spec,
            extra_body=p.extra_body if p else None,
        )

    defaults = config.agents.defaults
    provider.generation = GenerationSettings(
        temperature=defaults.temperature,
        max_tokens=defaults.max_tokens,
        reasoning_effort=defaults.reasoning_effort,
    )
    return provider


def provider_signature(
    config: Config,
    *,
    model_override: str | None = None,
    provider_override: str | None = None,
) -> tuple[object, ...]:
    """Return the config fields that affect the primary LLM provider."""
    model = model_override or config.agents.defaults.model
    defaults = config.agents.defaults
    p, provider_name, spec = _resolve_provider(config, model, provider_override)
    api_base = _api_base_for_provider(config, model, p, spec, provider_override=provider_override)
    return (
        model,
        provider_override or defaults.provider,
        provider_name,
        p.api_key if p else None,
        api_base,
        defaults.max_tokens,
        defaults.temperature,
        defaults.reasoning_effort,
        defaults.context_window_tokens,
    )


def build_provider_snapshot(
    config: Config,
    *,
    model_override: str | None = None,
    provider_override: str | None = None,
) -> ProviderSnapshot:
    model = model_override or config.agents.defaults.model
    return ProviderSnapshot(
        provider=make_provider(
            config,
            model_override=model_override,
            provider_override=provider_override,
        ),
        model=model,
        context_window_tokens=config.agents.defaults.context_window_tokens,
        signature=provider_signature(
            config,
            model_override=model_override,
            provider_override=provider_override,
        ),
    )


def build_gui_provider_snapshot(config: Config) -> ProviderSnapshot | None:
    """Create the optional GUI-specific provider snapshot from config.gui."""
    if config.gui is None:
        return None
    gui_model = config.gui.model or config.agents.defaults.model
    return build_provider_snapshot(
        config,
        model_override=gui_model,
        provider_override=config.gui.provider,
    )


def build_gui_s2_provider_snapshot(config: Config) -> ProviderSnapshot | None:
    """Create the optional GUI S2 provider snapshot from config.gui."""
    if config.gui is None or not config.gui.s2_model:
        return None
    return build_provider_snapshot(
        config,
        model_override=config.gui.s2_model,
        provider_override=config.gui.s2_provider,
    )


def load_provider_snapshot(config_path: Path | None = None) -> ProviderSnapshot:
    from nanobot.config.loader import load_config, resolve_config_env_vars

    return build_provider_snapshot(resolve_config_env_vars(load_config(config_path)))
