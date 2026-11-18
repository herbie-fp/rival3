from __future__ import annotations

import platform
import shutil
import subprocess
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

ROOT = Path(__file__).resolve().parent.parent


def native_target() -> tuple[str, str]:
    system = platform.system()
    arch = platform.machine().lower()
    arch = {"amd64": "x86_64", "arm64": "aarch64"}.get(arch, arch)
    if system == "Windows":
        return f"{arch}-pc-windows-msvc", "rival3_ffi.dll"
    if system == "Darwin":
        return f"{arch}-apple-darwin", "librival3_ffi.dylib"
    return f"{arch}-unknown-linux-gnu", "librival3_ffi.so"


class BuildWithRust(build_py):
    def run(self):
        subprocess.run(
            ["cargo", "build", "--release", "--manifest-path", str(ROOT / "rival3-ffi/Cargo.toml")],
            cwd=ROOT,
            check=True,
        )
        super().run()
        target, filename = native_target()
        source = ROOT / "rival3-ffi/target/release" / filename
        destination = Path(self.build_lib) / "rival3/private/native" / target
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination / filename)


try:
    from wheel.bdist_wheel import bdist_wheel

    class PlatformWheel(bdist_wheel):
        def finalize_options(self):
            super().finalize_options()
            self.root_is_pure = False

        def get_tag(self):
            _, _, platform_tag = super().get_tag()
            return "py3", "none", platform_tag

    setup(cmdclass={"build_py": BuildWithRust, "bdist_wheel": PlatformWheel})
except ImportError:
    setup(cmdclass={"build_py": BuildWithRust})
