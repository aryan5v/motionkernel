"""Generated graph-derived KernelSpec. Do not edit by hand."""
from pathlib import Path
from autokernel.specgen import spec_from_manifest
SPEC = spec_from_manifest(Path(__file__).with_name("manifest.json"))
