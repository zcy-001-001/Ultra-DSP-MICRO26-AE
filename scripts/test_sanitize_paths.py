#!/usr/bin/env python3
"""Unit smoke for AE path and host redaction patterns."""

from __future__ import annotations

import importlib.util
import io
import tarfile
import tempfile
import zipfile
from pathlib import Path


def load_sanitizer():
    path = Path(__file__).with_name("sanitize_paths.py")
    spec = importlib.util.spec_from_file_location("ae_sanitize_paths", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    sanitizer = load_sanitizer()
    hpc_home = "/hpc2" + "hdd/home/example_user"
    connect_home = "/home/" + "CONNECT/example_user"
    gpu_host = "gp" + "u3-1" + ".example.internal"
    gpu_node = "gp" + "u3" + "-1"
    gpu_node_in_run_name = "Jul21_12-37-17_" + gpu_node
    fpga_node = "fp" + "ga03"
    fpga_fqdn_in_run_name = "Jul21_12-37-17_" + "fp" + "ga04.example.internal"
    slurm_job = "SLURM_" + "JOB_ID=123456"
    windows_home = "C:" + r"\Users\example_user"
    private_model = "/data-" + "hdd/opt/models/model-a"
    private_scratch = "/scr" + "atch/example_user/run-a"
    generic_home = "/ho" + "me/example_user/run-b"
    private_workspace = "/work" + "space/private/run-c"
    private_work = "/wo" + "rk/private/run-d"
    private_mount = "/m" + "nt/private/run-e"
    samples = "\n".join(
        (
            f"model={hpc_home}/models/model-a",
            f"run={connect_home}/data/data/MICRO26/run-a",
            f"host={gpu_host}",
            f"node={gpu_node}",
            f"logging_dir=runs/{gpu_node_in_run_name}",
            f"build_host={fpga_node}",
            f"logging_dir=runs/{fpga_fqdn_in_run_name}",
            slurm_job,
            f"local={windows_home}\\Desktop\\MICRO\\AE-1305",
            f"model={private_model}",
            f"scratch={private_scratch}",
            f"generic_home={generic_home}",
        )
    )
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        fixture = root / "fixture.log"
        fixture.write_text(samples, encoding="utf-8")
        tar_path = root / "logs.tgz"
        tar_text = (
            f"model={private_model}-b\nrun={private_workspace}\n"
        ).encode()
        with tarfile.open(tar_path, "w:gz") as archive:
            member = tarfile.TarInfo("logs/eval.log")
            member.size = len(tar_text)
            archive.addfile(member, io.BytesIO(tar_text))
            binary = ("binary:" + private_model).encode()
            binary_member = tarfile.TarInfo("weights/payload.bin")
            binary_member.size = len(binary)
            archive.addfile(binary_member, io.BytesIO(binary))
        zip_path = root / "results.zip"
        with zipfile.ZipFile(zip_path, "w") as archive:
            archive.writestr(
                "results/result.json",
                f'{{"run":"{private_work}","mount":"{private_mount}"}}',
            )

        before = sanitizer.audit(root)
        assert any("logs.tgz!logs/eval.log" in item for item in before)
        assert any("results.zip!results/result.json" in item for item in before)
        changed, replacements = sanitizer.sanitize(root)
        assert changed == 3
        assert replacements == 16
        assert sanitizer.audit(root) == []
        sanitized = fixture.read_text(encoding="utf-8")
        assert hpc_home not in sanitized
        assert connect_home not in sanitized
        assert gpu_host not in sanitized
        assert gpu_node not in sanitized
        assert gpu_node_in_run_name not in sanitized
        assert fpga_node not in sanitized
        assert fpga_fqdn_in_run_name not in sanitized
        assert slurm_job not in sanitized
        assert windows_home not in sanitized
        assert private_model not in sanitized
        assert private_scratch not in sanitized
        assert generic_home not in sanitized
        assert sanitized.count("<REMOTE_HOME>") == 2
        assert sanitized.count("<REMOTE_WORKSPACE>") == 1
        assert sanitized.count("<REMOTE_HOST>") == 5
        assert "<REMOTE_STORAGE>/opt/models/model-a" in sanitized
        assert "<REMOTE_STORAGE>/example_user/run-a" in sanitized
        assert "SLURM_JOB_ID=<JOB_ID>" in sanitized
        assert "<AE_ROOT>" in sanitized
        with tarfile.open(tar_path, "r:gz") as archive:
            tar_sanitized = archive.extractfile("logs/eval.log").read().decode()
            assert ("/data-" + "hdd/") not in tar_sanitized
            assert ("/work" + "space/") not in tar_sanitized
            assert archive.extractfile("weights/payload.bin").read() == binary
        with zipfile.ZipFile(zip_path, "r") as archive:
            zip_sanitized = archive.read("results/result.json").decode()
            assert ("/wo" + "rk/") not in zip_sanitized
            assert ("/m" + "nt/") not in zip_sanitized
    print("SANITIZE_PATHS_TEST_PASS cases=16 archives=2")


if __name__ == "__main__":
    main()
