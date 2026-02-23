import argparse
import platform
import shutil
import subprocess
import sys

from ._emerge.projects.generate_project import generate_project

REPO_URL = "https://github.com/FennisRobert/EMerge.git"

SOLVERS = ["umfpack", "cudss", "dxf", "gerber", "aasds"]


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a subprocess command, streaming output to the terminal."""
    return subprocess.run(cmd, check=True, **kwargs)


def _pip(*args: str) -> None:
    _run([sys.executable, "-m", "pip", *args])


def cmd_new(args: argparse.Namespace) -> None:
    generate_project(args.projectname, args.filename)


def cmd_upgrade(args: argparse.Namespace) -> None:
    branch = args.branch or "main"
    url = f"git+{REPO_URL}@{branch}"
    print(f"Upgrading EMerge from branch '{branch}'...")
    _pip("install", "--upgrade", url)
    print("Upgrade complete.")


def cmd_install_solver(args: argparse.Namespace) -> None:
    solver = args.solver
    system = platform.system()       # 'Windows', 'Darwin', 'Linux'
    machine = platform.machine()     # 'AMD64', 'arm64', 'x86_64', ...

    if solver == "umfpack":
        _install_umfpack(system)

    elif solver in ("cudss", "cudss12"):
        _install_cudss(system)

    elif solver == "aasds":
        _install_aasds(system, machine)

    elif solver in ("dxf", "gerber"):
        _install_extras(solver)

    else:
        print(f"Unknown solver '{solver}'. Available: {', '.join(SOLVERS)}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Solver helpers
# ---------------------------------------------------------------------------

def _install_umfpack(system: str) -> None:
    if system == "Windows":
        conda = shutil.which("conda")
        if conda is None:
            print(
                "UMFPACK on Windows requires conda, but it was not found on PATH.\n"
                "Please install Miniconda or Anaconda and try again, or run:\n\n"
                "  conda install conda-forge::scikit-umfpack\n"
            )
            sys.exit(1)
        print("Installing scikit-umfpack via conda (conda-forge)...")
        _run([conda, "install", "--yes", "conda-forge::scikit-umfpack"])
    else:
        print("Installing scikit-umfpack via pip...")
        _pip("install", "scikit-umfpack")
    print("UMFPACK solver installed.")


def _install_cudss(system: str) -> None:
    if system != "Windows":
        print(
            "Warning: cuDSS is primarily targeted at Windows + NVIDIA GPUs. "
            "Proceeding anyway..."
        )
    packages = [
        "nvidia-cudss-cu12==0.5.0.16",
        "nvmath-python[cu12]==0.5.0",
        "cupy-cuda12x",
    ]
    print("Installing cuDSS solver dependencies...")
    _pip("install", *packages)
    print("cuDSS solver installed.")


def _install_aasds(system: str, machine: str) -> None:
    if system != "Darwin" or machine != "arm64":
        print(
            "The Apple Accelerate solver (emerge-aasds) is only supported on "
            "macOS with Apple Silicon (arm64).\n"
            f"Detected: {system} / {machine}"
        )
        sys.exit(1)
    print("Installing emerge-aasds (Apple Accelerate solver)...")
    _pip("install", "emerge-aasds")
    print("Apple Accelerate solver installed.")


def _install_extras(extra: str) -> None:
    package_map = {
        "dxf": "ezdxf",
        "gerber": "pygerber",
    }
    package = package_map[extra]
    print(f"Installing {extra} dependency ({package})...")
    _pip("install", package)
    print(f"{extra} dependency installed.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="EMerge FEM Solver CLI")
    subparsers = parser.add_subparsers(dest="command")

    # --- new ---
    new_parser = subparsers.add_parser("new", help="Create a new project")
    new_parser.add_argument("projectname", type=str, help="Name of the project directory")
    new_parser.add_argument("filename", type=str, help="Base name for files")

    # --- upgrade ---
    upgrade_parser = subparsers.add_parser(
        "upgrade", help="Upgrade EMerge to the latest version from git"
    )
    upgrade_parser.add_argument(
        "--branch", "-b",
        type=str,
        default=None,
        metavar="BRANCH",
        help="Git branch to install from (default: main)",
    )

    # --- install-solver ---
    solver_parser = subparsers.add_parser(
        "install-solver", help="Install an optional solver backend"
    )
    solver_parser.add_argument(
        "solver",
        choices=SOLVERS,
        help=(
            "Solver to install: "
            "umfpack (Linux/macOS via pip, Windows via conda), "
            "cudss / cudss12 (Windows + NVIDIA GPU), "
            "aasds (macOS Apple Silicon), "
            "dxf, gerber (file format extras)"
        ),
    )

    args = parser.parse_args()

    if args.command == "new":
        cmd_new(args)
    elif args.command == "upgrade":
        cmd_upgrade(args)
    elif args.command == "install-solver":
        cmd_install_solver(args)
    else:
        parser.print_help()