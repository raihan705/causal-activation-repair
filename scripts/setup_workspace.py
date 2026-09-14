#!/usr/bin/env python3
"""Create the logical directory layout expected by the released workflows."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


RELEASE_ROOT = Path(__file__).resolve().parents[1]


def copy_tree(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, dirs_exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workdir",
        type=Path,
        required=True,
        help="Empty or dedicated reconstruction directory; never point this at the release repository.",
    )
    args = parser.parse_args()
    workdir = args.workdir.expanduser().resolve()
    if workdir == RELEASE_ROOT or RELEASE_ROOT in workdir.parents:
        raise SystemExit("The reconstruction workspace must be outside the release repository.")
    workdir.mkdir(parents=True, exist_ok=True)

    copy_tree(RELEASE_ROOT / "scripts" / "model1" / "core", workdir / "phases")
    for phase_directory in sorted((RELEASE_ROOT / "scripts" / "model1" / "revision").iterdir()):
        if phase_directory.is_dir():
            copy_tree(
                phase_directory,
                workdir / "revision" / "model1" / phase_directory.name / "scripts",
            )
    copy_tree(
        RELEASE_ROOT / "scripts" / "model1" / "utility_routes",
        workdir / "revision" / "model1" / "routewise_bstar_utility" / "scripts",
    )
    copy_tree(RELEASE_ROOT / "scripts" / "model2", workdir / "revision" / "model2" / "phase21" / "scripts")
    copy_tree(RELEASE_ROOT / "scripts" / "model3", workdir / "revision" / "model3" / "phase22" / "scripts")
    copy_tree(RELEASE_ROOT / "configs" / "model1", workdir / "configs")

    for relative in (
        "data/processed",
        "data/cyberseceval",
        "data/activations/raw",
        "data/activations/latent",
        "outputs",
        "revision/model1/routewise_bstar_utility/outputs",
        "revision/model2/phase21/outputs",
        "revision/model3/phase22/outputs",
    ):
        (workdir / relative).mkdir(parents=True, exist_ok=True)

    print(f"Prepared reconstruction workspace: {workdir}")
    print(f"Set CAR_ROOT={workdir} when running legacy Model 1 entry points.")


if __name__ == "__main__":
    main()
