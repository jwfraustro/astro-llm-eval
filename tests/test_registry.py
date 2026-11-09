"""Every task package must agree with its registry entry."""

from __future__ import annotations

import importlib

import pytest

from tasks import registry


@pytest.mark.parametrize("name", sorted(registry.TASKS))
def test_registry_entry_matches_package(name: str):
    info = registry.get(name)
    assert info.directory.is_dir()
    assert info.variants_file.is_file()

    # The variants file loads with the registered class.
    variants_mod = importlib.import_module(f"tasks.{name}.variants")
    loaded = variants_mod.load_variants(info.variants_file)
    assert loaded and all(isinstance(v, info.variant_cls) for v in loaded)
    assert all(v.id.startswith(f"{name}-") for v in loaded)

    # The registered scorer names are real scorer factories in the package.
    scorer_mod = importlib.import_module(f"tasks.{name}.scorer")
    assert callable(getattr(scorer_mod, info.outcome_scorer))
    if info.process_scorer:
        assert callable(getattr(scorer_mod, info.process_scorer))

    # The @task function exists under the task's own name.
    task_mod = importlib.import_module(f"tasks.{name}.task")
    assert callable(getattr(task_mod, name))

    assert hasattr(info.client_factory, "healthcheck")
    assert issubclass(info.service_error, Exception)


def test_for_scorers_and_unknown():
    assert registry.for_scorers({"adql_verify", "other"}).name == "adql_translate"
    assert registry.for_scorers({"nothing"}) is None
    with pytest.raises(KeyError):
        registry.get("not_a_task")
