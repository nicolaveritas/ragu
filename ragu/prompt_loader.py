"""Generic YAML+Jinja prompt loader, shared by any `prompts/` folder in the repo."""

from functools import lru_cache
from pathlib import Path

import yaml
from jinja2 import Template


@lru_cache
def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def render_prompt(prompts_dir: Path, name: str, **kwargs) -> str:
    """Render the `template` field of <prompts_dir>/<name>.yaml with kwargs."""
    spec = _load(prompts_dir / f"{name}.yaml")
    return Template(spec["template"]).render(**kwargs).strip()
