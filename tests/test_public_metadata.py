"""Guards against drift in the things a public release gets judged on.

Product metadata, package metadata, documentation links and the compatibility
import namespace all drift silently: nothing fails when a version is bumped in
one file and not the other, when a doc grows a link to a file that was later
renamed, or when the alias namespace stops resolving. Each of those is
embarrassing in public and cheap to catch here.

Everything in this module is CPU-only and reads the repository, not a built
distribution, so it runs in the ordinary suite.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

# tomllib is 3.11+; the project supports 3.10, where tomli is the backport.
try:  # pragma: no cover - exercised by whichever interpreter runs the suite
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:  # pragma: no cover
        tomllib = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The product name. The import namespace deliberately differs; see
#: docs/NAMESPACE_MIGRATION.md.
PRODUCT_NAME = "MotionKernel"
DISTRIBUTION_NAME = "motionkernel"
CANONICAL_NAMESPACE = "motionkernel"
COMPATIBILITY_NAMESPACE = "autokernel"

#: Upstream identity that must never be stripped: removing it would breach the
#: MIT license, not merely lose provenance.
UPSTREAM_COPYRIGHT = "Copyright (c) 2026 RightNow AI"
UPSTREAM_PROJECT_URL = "https://github.com/RightNow-AI/autokernel"

#: Root documents that face the public directly.
PUBLIC_DOCS = (
    "README.md",
    "DOWNSTREAM.md",
    "ROADMAP.md",
    "CONTRIBUTING.md",
    "CHANGELOG.md",
    "SECURITY.md",
    "PROVENANCE.md",
)


@pytest.fixture(scope="module")
def pyproject() -> dict:
    if tomllib is None:  # pragma: no cover - only on 3.10 without tomli
        pytest.skip("needs tomllib (3.11+) or the tomli backport; see the dev extra")
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


# -- package metadata ---------------------------------------------------


def test_distribution_is_named_for_the_product(pyproject: dict) -> None:
    assert pyproject["project"]["name"] == DISTRIBUTION_NAME


def test_package_and_distribution_versions_agree(pyproject: dict) -> None:
    """These lived in two files and disagreed (0.1.0 vs 1.0.0) before this test."""
    sys.path.insert(0, str(REPO_ROOT))
    try:
        import autokernel
    finally:
        sys.path.pop(0)
    assert pyproject["project"]["version"] == autokernel.__version__, (
        "pyproject.toml version and autokernel.__version__ have drifted; "
        "they are one release identity, not two"
    )


def test_both_namespaces_are_packaged(pyproject: dict) -> None:
    """A wheel missing either namespace breaks a documented import path."""
    packages = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    assert COMPATIBILITY_NAMESPACE in packages
    assert CANONICAL_NAMESPACE in packages


def test_console_script_resolves_to_a_real_entry_point(pyproject: dict) -> None:
    target = pyproject["project"]["scripts"][DISTRIBUTION_NAME]
    module, _, symbol = target.partition(":")
    path = REPO_ROOT / Path(*module.split(".")).with_suffix(".py")
    assert path.is_file(), f"console script points at missing module {module}"
    assert re.search(rf"^def {re.escape(symbol)}\b", path.read_text(), re.M), (
        f"console script points at missing symbol {target}"
    )


def test_license_and_readme_are_declared(pyproject: dict) -> None:
    assert pyproject["project"]["license"]["file"] == "LICENSE"
    assert pyproject["project"]["readme"] == "README.md"


def test_project_urls_point_at_files_that_exist(pyproject: dict) -> None:
    """A dead link in package metadata is visible on the PyPI page forever."""
    for name, url in pyproject["project"]["urls"].items():
        match = re.search(r"/blob/main/(.+)$", url)
        if match is None:
            continue
        target = REPO_ROOT / match.group(1)
        assert target.is_file(), f"project URL {name} names missing file {match.group(1)}"


# -- licensing and provenance -------------------------------------------


def test_upstream_copyright_is_preserved() -> None:
    """Required by the MIT license, not merely polite."""
    license_text = (REPO_ROOT / "LICENSE").read_text(encoding="utf-8")
    assert UPSTREAM_COPYRIGHT in license_text
    assert "MIT License" in license_text


def test_provenance_inventory_exists_and_is_generated() -> None:
    provenance = (REPO_ROOT / "PROVENANCE.md").read_text(encoding="utf-8")
    assert "scripts/provenance_inventory.py" in provenance
    for bucket in ("Unchanged from upstream", "Modified descendants", "MotionKernel-original"):
        assert bucket in provenance


def test_fork_provenance_is_disclosed() -> None:
    """Removing the attribution would misrepresent the project's origin."""
    downstream = (REPO_ROOT / "DOWNSTREAM.md").read_text(encoding="utf-8")
    assert UPSTREAM_PROJECT_URL in downstream
    assert UPSTREAM_COPYRIGHT in downstream


# -- product language ---------------------------------------------------


@pytest.mark.parametrize("name", PUBLIC_DOCS)
def test_public_docs_lead_with_the_product_name(name: str) -> None:
    text = (REPO_ROOT / name).read_text(encoding="utf-8")
    assert PRODUCT_NAME in text, f"{name} never names the product"


def test_readme_does_not_present_upstream_releases_as_our_own() -> None:
    """The README once carried upstream's v1.x release notes under its own
    'Changelog' heading, which reads as MotionKernel's release history."""
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    for stale in ("AMD ROCm GPU support: MI300X", "Initial release: Triton kernel"):
        assert stale not in readme, (
            "README repeats upstream release notes; link CHANGELOG.md instead"
        )


def test_autokernel_is_described_as_a_compatibility_namespace() -> None:
    """Anyone seeing the import must be able to find out what it means."""
    for name in ("README.md", "DOWNSTREAM.md"):
        text = (REPO_ROOT / name).read_text(encoding="utf-8")
        assert "compatibility namespace" in text.lower(), (
            f"{name} does not explain that {COMPATIBILITY_NAMESPACE} is a "
            "compatibility namespace rather than the product name"
        )


# -- documentation links ------------------------------------------------


def _relative_links(text: str) -> list[str]:
    links = re.findall(r"\[[^\]]*\]\(([^)]+)\)", text)
    out = []
    for link in links:
        target = link.split("#", 1)[0].strip()
        if not target or target.startswith(("http://", "https://", "mailto:")):
            continue
        out.append(target)
    return out


@pytest.mark.parametrize(
    "name",
    PUBLIC_DOCS + tuple(f"docs/{p.name}" for p in sorted((REPO_ROOT / "docs").glob("*.md"))),
)
def test_relative_documentation_links_resolve(name: str) -> None:
    source = REPO_ROOT / name
    text = source.read_text(encoding="utf-8")
    missing = [
        link
        for link in _relative_links(text)
        if not (source.parent / link).resolve().exists()
    ]
    assert not missing, f"{name} links to missing file(s): {missing}"


# -- compatibility imports ----------------------------------------------


def test_compatibility_namespace_still_imports() -> None:
    """Every generated spec.py ever emitted imports this path."""
    sys.path.insert(0, str(REPO_ROOT))
    try:
        import autokernel.specgen

        assert hasattr(autokernel.specgen, "spec_from_manifest")
    finally:
        sys.path.pop(0)




def test_generated_specs_still_import_a_shipped_namespace() -> None:
    """specgen emits an import into every artifact; artifacts are hash-verified,
    so changing it invalidates bundles that already exist."""
    generator = (REPO_ROOT / "autokernel" / "specgen" / "generator.py").read_text(
        encoding="utf-8"
    )
    emitted = re.findall(r"from (\w+)\.specgen import", generator)
    assert emitted, "no emitted spec import found; update this guard"
    for namespace in set(emitted):
        assert namespace in {COMPATIBILITY_NAMESPACE, CANONICAL_NAMESPACE}, (
            f"generated specs import {namespace!r}, which this release does not ship"
        )


# -- support claims -----------------------------------------------------


def test_support_status_page_exists_and_defines_its_levels() -> None:
    text = (REPO_ROOT / "docs" / "SUPPORT_STATUS.md").read_text(encoding="utf-8")
    for level in ("Proven", "Validated (isolated)", "In progress", "Target"):
        assert level in text


def test_cosmos_is_not_claimed_as_supported() -> None:
    """Cosmos has no published end-to-end evidence, so it must not read as
    proven or validated. A run being under way is not evidence."""
    text = (REPO_ROOT / "docs" / "SUPPORT_STATUS.md").read_text(encoding="utf-8")
    row = next(line for line in text.splitlines() if line.startswith("| Cosmos"))
    assert any(level in row for level in ("Candidate", "In progress", "Target")), (
        f"Cosmos row claims more than the evidence supports: {row}"
    )
    assert "Proven" not in row and "Validated" not in row


def test_support_levels_include_candidate() -> None:
    text = (REPO_ROOT / "docs" / "SUPPORT_STATUS.md").read_text(encoding="utf-8")
    assert "**Candidate**" in text


def test_proven_rows_carry_evidence_links() -> None:
    """Every Proven entry in a support table must link its evidence.

    Only four-column rows are inspected: those are the support tables. The
    Levels table that *defines* "Proven" has two columns and is not a claim.
    """
    text = (REPO_ROOT / "docs" / "SUPPORT_STATUS.md").read_text(encoding="utf-8")
    checked = 0
    for line in text.splitlines():
        if not line.startswith("|") or "**Proven**" not in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 4:
            continue
        checked += 1
        assert "](" in cells[-1], f"Proven row without an evidence link: {line}"
    assert checked, "no Proven support rows found; update this guard"


# -- canonical namespace ------------------------------------------------
#
# The canonical namespace is plain re-export modules, not an import hook. An
# earlier revision used a sys.meta_path finder; it preserved module identity
# but was invisible to type checkers, which is the one property the namespace
# being recommended most needs.


def test_canonical_namespace_shares_class_identity() -> None:
    """Module identity is not shared and does not need to be; class identity
    is what makes isinstance work across the two namespaces."""
    sys.path.insert(0, str(REPO_ROOT))
    try:
        import autokernel.specs
        import autokernel.verification
        import motionkernel
        import motionkernel.specs
        import motionkernel.verification

        assert motionkernel.specs.KernelSpec is autokernel.specs.KernelSpec
        assert motionkernel.specs.Tolerance is autokernel.specs.Tolerance
        assert (
            motionkernel.verification.ParityPolicy
            is autokernel.verification.ParityPolicy
        )
        assert motionkernel.__version__ == autokernel.__version__
    finally:
        sys.path.pop(0)


def test_isinstance_holds_across_namespaces() -> None:
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from autokernel.specs import Tolerance as A
        from motionkernel.specs import Tolerance as M

        assert isinstance(A(atol=1e-3, rtol=1e-3), M)
        assert isinstance(M(atol=1e-3, rtol=1e-3), A)
    finally:
        sys.path.pop(0)


def test_canonical_namespace_uses_no_import_hook() -> None:
    """No sys.meta_path mutation, so importing it cannot affect unrelated
    imports elsewhere in the process."""
    source = (REPO_ROOT / "motionkernel" / "__init__.py").read_text(encoding="utf-8")
    # The prose explains why there is no hook, so match the call, not the word.
    assert "sys.meta_path" not in source.replace("``sys.meta_path``", "")
    assert "meta_path.append" not in source
    assert "class _AliasFinder" not in source


def test_every_public_subpackage_is_re_exported() -> None:
    """A subpackage present under autokernel but missing here is a silent gap
    in the canonical namespace."""
    expected = {
        path.name
        for path in (REPO_ROOT / COMPATIBILITY_NAMESPACE).iterdir()
        if path.is_dir() and (path / "__init__.py").is_file() and not path.name.startswith("_")
    }
    shipped = {
        path.stem
        for path in (REPO_ROOT / CANONICAL_NAMESPACE).glob("*.py")
        if path.stem != "__init__"
    }
    missing = expected - shipped - {"workloads"}
    assert not missing, f"canonical namespace is missing: {sorted(missing)}"


def test_both_packages_ship_a_py_typed_marker() -> None:
    """Without it, an installed wheel gives downstream users no types at all."""
    for namespace in (COMPATIBILITY_NAMESPACE, CANONICAL_NAMESPACE):
        marker = REPO_ROOT / namespace / "py.typed"
        assert marker.is_file(), f"{namespace} has no py.typed marker"


def test_the_namespace_tradeoff_is_documented() -> None:
    text = (REPO_ROOT / "docs" / "NAMESPACE_MIGRATION.md").read_text(encoding="utf-8")
    assert "What the re-export gives up" in text
    assert "Deep module paths" in text


def test_docs_do_not_claim_artifacts_pin_the_namespace() -> None:
    """They do not. A packaged bundle is candidate.py, entry.py and
    manifest.json; none of them import the package."""
    for name in ("README.md", "DOWNSTREAM.md", "docs/NAMESPACE_MIGRATION.md"):
        text = (REPO_ROOT / name).read_text(encoding="utf-8")
        assert "invalidate the manifest of every artifact" not in text
