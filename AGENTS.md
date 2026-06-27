# AGENTS.md

## Commands

```bash
# Install (editable, with common extras)
pip install -e ".[transformers,ray,test]"

# Lint (must pass before PR)
pre-commit run --all-files

# Run all tests (requires GPU)
pytest tests

# Run a single test file
pytest tests/infra/test_infra_graph.py

# Run a single test
pytest tests/infra/test_infra_graph.py::TestClassName::test_method
```

## Lint & Formatting

- **Formatter:** yapf (pep8, 120-col), **not** black or ruff format
- **Import sort:** isort (120-col, known_first_party=twinkle)
- **Linter:** flake8 (120-col, ignores F401/F403/F405/F821/W503/E251/W504/E126/E125)
- **String quotes:** single quotes enforced by `double-quote-string-fixer`
- **Line endings:** LF only
- Pre-commit **excludes** `cookbook/`, `client_tools/`, `src/twinkle_client/`, `tests/`, `examples/`

## Architecture

- **PyPI package:** `twinkle-kit`; Python 3.11–3.12
- **Three packages** under `src/`: `twinkle` (core), `twinkle_client` (remote client), `twinkle_agentic` (async RL)
- **Lazy imports:** `src/twinkle/__init__.py` uses `_LazyModule`; import public API from `twinkle` directly
- **Distributed modes:** torchrun (local), Ray, HTTP server — selected via `twinkle.initialize(mode=...)` or `TWINKLE_MODE` env
- **Backends:** Transformers (FSDP2) and Megatron (TP/PP/CP); Megatron requires separate install via `INSTALL_MEGATRON.sh`
- **Remote decorators:** `remote_class()` / `remote_function(dispatch, execute, collect)` wrap classes/methods for Ray placement
- **Device topology:** `DeviceMesh` / `DeviceGroup`; visualize with `twinkle.infra.get_device_placement()`
- **Hub routing:** `HubOperation` dispatches to HF or ModelScope by `hf://` or `ms://` prefix
- **Server entry point:** `twinkle-server` CLI → `twinkle.server.cli:main`
- **Cookbook:** `cookbook/` has runnable example scripts for each training mode (not unit-tested)

## Conventions

- Variable names: `snake_case`; classes: `PascalCase`; 4-space indent
- Remote plugin code requires `trust_remote_code()` guard — never load untrusted adapter configs
- Env-driven ranks: `RANK`, `WORLD_SIZE`, `LOCAL_RANK` must be set for torchrun
- Multi-LoRA: FSDP unsupported (`fsdp_world_size == 1` enforced); base weights are frozen, only adapter params train
- CI runs in Docker on self-hosted GPU runners; NPU CI runs on Ascend 910B
