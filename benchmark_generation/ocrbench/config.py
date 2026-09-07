"""Configuration loading and deterministic weighted selection.

All tunable behaviour lives in configs/*.yaml. This module loads those files,
validates the parts the pipeline depends on, and exposes them as a single
`BenchmarkConfig`. No pipeline module reads YAML directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from .hashing import config_hash


class ConfigError(ValueError):
    """Raised when a configuration file is missing required structure."""


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"configuration file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a YAML mapping at the top level")
    return data


def _require(mapping: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"missing required key '{key}' in {where}")
    return mapping[key]


# ---------------------------------------------------------------------------
# Deterministic weighted choice
# ---------------------------------------------------------------------------
def weighted_choice(options: Sequence[Mapping[str, Any]], rng: np.random.Generator) -> Any:
    """Pick one `value` from a list of {value, weight} entries.

    Deterministic for a given RNG state. Entries without an explicit weight
    default to 1. Raises on an empty pool rather than returning None.
    """
    if not options:
        raise ConfigError("cannot choose from an empty option pool")
    weights = np.array([float(o.get("weight", 1.0)) for o in options], dtype=np.float64)
    if weights.sum() <= 0:
        raise ConfigError("option pool has non-positive total weight")
    weights /= weights.sum()
    idx = int(rng.choice(len(options), p=weights))
    entry = options[idx]
    if "value" not in entry:
        raise ConfigError(f"option pool entry lacks a 'value' key: {entry!r}")
    return entry["value"]


def weighted_choice_entry(
    options: Sequence[Mapping[str, Any]], rng: np.random.Generator
) -> Mapping[str, Any]:
    """As `weighted_choice`, but returns the whole entry (keeps `name`, etc.)."""
    if not options:
        raise ConfigError("cannot choose from an empty option pool")
    weights = np.array([float(o.get("weight", 1.0)) for o in options], dtype=np.float64)
    if weights.sum() <= 0:
        raise ConfigError("option pool has non-positive total weight")
    weights /= weights.sum()
    return options[int(rng.choice(len(options), p=weights))]


# ---------------------------------------------------------------------------
# Distortion configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DistortionSpec:
    """One atomic distortion and its per-severity parameter sets."""

    name: str
    family: str
    output_dir: str
    description: str
    severities: Mapping[str, Mapping[str, Any]]

    def params(self, severity: str) -> Mapping[str, Any]:
        if severity not in self.severities:
            raise ConfigError(
                f"distortion '{self.name}' has no severity '{severity}' "
                f"(available: {sorted(self.severities)})"
            )
        return self.severities[severity]

    @property
    def levels(self) -> list[str]:
        return list(self.severities)


@dataclass(frozen=True)
class ProfileStep:
    distortion: str
    severity: str


@dataclass(frozen=True)
class ProfileSpec:
    """A fixed, reusable combination of atomic distortions."""

    name: str
    output_dir: str
    description: str
    steps: tuple[ProfileStep, ...]


@dataclass
class BenchmarkConfig:
    """Everything the pipeline needs, loaded from configs/."""

    distortions: dict[str, DistortionSpec]
    profiles: dict[str, ProfileSpec]
    readability: dict[str, Any]
    layout: dict[str, Any]
    allocation: dict[str, Any]
    raw_distortions: dict[str, Any] = field(repr=False, default_factory=dict)
    paths: dict[str, str] = field(repr=False, default_factory=dict)

    # -- derived ----------------------------------------------------------
    @property
    def hash(self) -> str:
        """Stable hash over all configuration that affects output bytes."""
        return config_hash(self.raw_distortions, self.layout, self.allocation)

    def distortion(self, name: str) -> DistortionSpec:
        if name not in self.distortions:
            raise ConfigError(
                f"unknown distortion '{name}' (available: {sorted(self.distortions)})"
            )
        return self.distortions[name]

    def profile(self, name: str) -> ProfileSpec:
        if name not in self.profiles:
            raise ConfigError(f"unknown profile '{name}' (available: {sorted(self.profiles)})")
        return self.profiles[name]

    def family_of(self, distortion_name: str) -> str:
        return self.distortion(distortion_name).family

    def output_dir_for(self, distortion_name: str) -> str:
        return self.distortion(distortion_name).output_dir


def _expand_mapping_profile(
    name: str,
    mapping: Mapping[str, str],
    aliases: Mapping[str, str],
    order: Sequence[str],
    known: Mapping[str, Any],
) -> tuple[ProfileStep, ...]:
    """Expand the `{alias: severity}` shorthand into an ordered step list."""
    resolved: dict[str, str] = {}
    for alias, severity in mapping.items():
        target = aliases.get(alias, alias)
        if target not in known:
            raise ConfigError(
                f"profile '{name}' references unknown distortion '{alias}'"
                f" (resolved to '{target}')"
            )
        if target in resolved:
            raise ConfigError(
                f"profile '{name}' maps two aliases onto the same distortion '{target}'"
            )
        resolved[target] = str(severity)

    ordered = [d for d in order if d in resolved]
    # Anything not named in mapping_expansion_order is appended in sorted order
    # so expansion stays deterministic even if the order list is incomplete.
    ordered += sorted(d for d in resolved if d not in ordered)
    return tuple(ProfileStep(distortion=d, severity=resolved[d]) for d in ordered)


def load_config(
    distortions_path: Path,
    layout_path: Path,
    allocation_path: Path,
) -> BenchmarkConfig:
    """Load and validate all configuration files."""
    raw_d = _load_yaml(distortions_path)
    layout = _load_yaml(layout_path)
    allocation = _load_yaml(allocation_path)

    raw_specs = _require(raw_d, "distortions", str(distortions_path))
    if not isinstance(raw_specs, dict) or not raw_specs:
        raise ConfigError(f"{distortions_path}: 'distortions' must be a non-empty mapping")

    distortions: dict[str, DistortionSpec] = {}
    for name, body in raw_specs.items():
        if not isinstance(body, dict):
            raise ConfigError(f"distortion '{name}' must be a mapping")
        severities = _require(body, "severities", f"distortion '{name}'")
        if not isinstance(severities, dict) or not severities:
            raise ConfigError(f"distortion '{name}': 'severities' must be a non-empty mapping")
        for level, params in severities.items():
            if not isinstance(params, dict):
                raise ConfigError(
                    f"distortion '{name}' severity '{level}' must be a mapping of parameters"
                )
        distortions[name] = DistortionSpec(
            name=name,
            family=str(body.get("family", name)),
            output_dir=str(body.get("output_dir", body.get("family", name))),
            description=str(body.get("description", "")).strip(),
            severities=severities,
        )

    aliases = raw_d.get("profile_aliases", {}) or {}
    order = raw_d.get("mapping_expansion_order", []) or []
    profiles: dict[str, ProfileSpec] = {}
    for name, body in (raw_d.get("profiles") or {}).items():
        if not isinstance(body, dict):
            raise ConfigError(f"profile '{name}' must be a mapping")
        if "steps" in body and "mapping" in body:
            raise ConfigError(f"profile '{name}' declares both 'steps' and 'mapping'; pick one")
        if "steps" in body:
            steps_raw = body["steps"]
            if not isinstance(steps_raw, list) or not steps_raw:
                raise ConfigError(f"profile '{name}': 'steps' must be a non-empty list")
            steps = []
            for step in steps_raw:
                dname = _require(step, "distortion", f"profile '{name}' step")
                sev = _require(step, "severity", f"profile '{name}' step")
                if dname not in distortions:
                    raise ConfigError(f"profile '{name}' references unknown distortion '{dname}'")
                distortions[dname].params(str(sev))  # validates severity exists
                steps.append(ProfileStep(distortion=str(dname), severity=str(sev)))
            step_tuple = tuple(steps)
        elif "mapping" in body:
            step_tuple = _expand_mapping_profile(
                name, body["mapping"], aliases, order, distortions
            )
            for step in step_tuple:
                distortions[step.distortion].params(step.severity)
        else:
            raise ConfigError(f"profile '{name}' must declare either 'steps' or 'mapping'")

        profiles[name] = ProfileSpec(
            name=name,
            output_dir=str(body.get("output_dir", "combined")),
            description=str(body.get("description", "")).strip(),
            steps=step_tuple,
        )

    # Cross-validate the allocation file against the distortion catalogue now,
    # so a typo fails at startup rather than halfway through a long run.
    for bucket, body in (allocation.get("buckets") or {}).items():
        for dname in body.get("distortions", []):
            if dname not in distortions:
                raise ConfigError(
                    f"allocation bucket '{bucket}' references unknown distortion '{dname}'"
                )
    for pname in ((allocation.get("combined") or {}).get("profiles") or {}):
        if pname not in profiles:
            raise ConfigError(f"allocation references unknown profile '{pname}'")

    return BenchmarkConfig(
        distortions=distortions,
        profiles=profiles,
        readability=dict(raw_d.get("readability", {}) or {}),
        layout=layout,
        allocation=allocation,
        raw_distortions=raw_d,
        paths={
            "distortions": str(distortions_path),
            "layout": str(layout_path),
            "allocation": str(allocation_path),
        },
    )
