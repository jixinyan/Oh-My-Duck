# Modified by Oh My Duck: project-owned namespace, registration and module layout.
"""Publish a policy to the Hugging Face Hub in the shape the microduck daemon loads.

`uv run publish` — see :mod:`oh_my_duck.rl.artifacts.publish.cli`. The manifest contract is
`docs/policy-manifest.md` in the `pollen-robotics/microduck` repo (schema 2); the constants and
builder live in :mod:`oh_my_duck.rl.artifacts.publish.manifest`.
"""
