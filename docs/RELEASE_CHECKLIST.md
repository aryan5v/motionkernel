# Release checklist

Work top to bottom. Anything that cannot be ticked blocks the release; there is
no "we will fix it in a patch" row.

## 1. Identity and metadata

- [ ] `pyproject.toml` version matches `autokernel.__version__`
      (`pytest tests/test_public_metadata.py` enforces this).
- [ ] `CHANGELOG.md` has a dated section for this version, and the
      `Unreleased (downstream)` section is empty or carried forward.
- [ ] Upstream release history below the downstream sections is untouched.
- [ ] `LICENSE` still carries the upstream MIT notice and copyright.
- [ ] `PROVENANCE.md` regenerated:
      `python scripts/provenance_inventory.py --check` passes.

## 2. Claims

- [ ] Every "supported" in public docs resolves to a row in
      [SUPPORT_STATUS.md](SUPPORT_STATUS.md) with a linked evidence file.
- [ ] No performance number appears without GPU, software stack, methodology
      and baseline, per [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
- [ ] Models with no end-to-end evidence are listed as *in progress* or
      *target*, never as supported.
- [ ] The scope limits on the V1 proof (one workload, one GPU, one model, one
      artifact) are still stated wherever the headline number appears.

## 3. Code

- [ ] Full CPU suite passes: `pytest -q`.
- [ ] `ruff check .` clean on changed files.
- [ ] Compatibility namespace still imports and still aliases:
      `python -c "import motionkernel, autokernel; assert motionkernel.specs is autokernel.specs"`.
- [ ] Console script resolves: `motionkernel --help`.
- [ ] No GPU is initialized at import time for any package module.

## 4. Artifacts

- [ ] The artifact manifest schema version is unchanged, or the loader accepts
      both the old and new versions with a test for each.
- [ ] Generated `spec.py` files still import a namespace this release ships.
- [ ] A previously promoted bundle still verifies against this build.

## 5. Packaging

- [ ] `python -m build` produces a wheel and sdist.
- [ ] The wheel contains both `autokernel/` and `motionkernel/`.
- [ ] The sdist contains `LICENSE`, `README.md`, `CHANGELOG.md` and
      `PROVENANCE.md`.
- [ ] Installing the wheel into a clean environment allows
      `from motionkernel.specs import KernelSpec` and `motionkernel --help`.

## 6. Documentation

- [ ] Every relative link in the root Markdown files resolves
      (`pytest tests/test_public_metadata.py` enforces this).
- [ ] `README.md`, `DOWNSTREAM.md`, `ROADMAP.md`, `CONTRIBUTING.md`,
      `SECURITY.md` reviewed for stale product language.
- [ ] `docs/SUPPORT_STATUS.md` levels match reality.

## 7. Tag and publish

- [ ] Tag `v<version>`, annotated, on a commit that passed the above.
- [ ] Release notes summarise the CHANGELOG section and link the evidence for
      any claim they repeat.
- [ ] History is not squashed or rewritten — the fork's commit lineage is part
      of the provenance record.
