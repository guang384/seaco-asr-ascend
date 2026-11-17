#!/usr/bin/env python3
"""NPU 环境预检（部署前跑）：只回答「设备发现 + 驱动 + 基础算子能不能走通」。

stdlib 档（任何机器可跑，无需 torch）：/dev 节点、npu-smi、driver 挂载、/dev/shm。
--deep 档（需 torch_npu 环境，如镜像内）：torch.zeros + synchronize 真探 + 显存信息。

用法：
    python3 npu_probe_check.py            # stdlib 档
    python3 npu_probe_check.py --deep     # 附加 torch_npu 实探
退出码：0 全通过（WARN 不算失败）；1 存在 ERROR（按提示处理后重跑）。
移植自 kokoro-tts-ascend/scripts/npu_probe_check.py（ASR 口径裁剪）。
"""
import argparse
import os
import subprocess
import sys

OK, WARN, ERR = "OK  ", "WARN", "ERR "
results = []


def add(level, name, detail=""):
    results.append((level, name, detail))
    print(f"[{level}] {name}" + (f"  | {detail}" if detail else ""))


def run(cmd, timeout=15):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except FileNotFoundError:
        return -1, "", "not found"
    except subprocess.TimeoutExpired:
        return -2, "", f"timeout {timeout}s"


def check_devices():
    """驱动层可见性：容器挂载的 4 个设备节点缺一不可。"""
    for dev in ("/dev/davinci_manager", "/dev/devmm_svm", "/dev/hisi_hdc"):
        present = os.path.exists(dev)
        add(OK if present else ERR, f"device {dev}",
            "present" if present else
            "missing: 容器挂载缺设备（--device /dev/...，见 deploy.sh npu_mounts）")
    devs = [d for d in os.listdir("/dev") if d.startswith("davinci")
            and d != "davinci_manager"] if os.path.isdir("/dev") else []
    add(OK if devs else ERR, "davinci 节点", ",".join(sorted(devs)) or "未发现")


def check_driver_mount():
    ok = os.path.isdir("/usr/local/Ascend/driver")
    add(OK if ok else ERR, "/usr/local/Ascend/driver",
        "mounted" if ok else "missing: 容器需 -v /usr/local/Ascend/driver（deploy.sh 已带）")


def check_smi():
    """驱动存活：npu-smi info 能跑且解析出卡号。"""
    rc, out, err = run(["npu-smi", "info"])
    if rc != 0:
        add(ERR, "npu-smi info", f"rc={rc} {err[:120]}（驱动未就绪？）")
        return
    ids = []
    for line in out.splitlines():
        if "Process id" in line:
            break
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            ids.append(int(parts[1]))
    add(OK if ids else WARN, "npu-smi 卡号",
        f"解析到芯片 {ids}" if ids else "未解析到（表格格式差异？不影响 torch_npu 实探）")


def check_shm():
    """音频 SHM 块的 tmpfs 容量：过小写满即 SIGBUS。"""
    try:
        st = os.statvfs("/dev/shm")
        mb = st.f_frsize * st.f_blocks / 2**20
    except (OSError, AttributeError):
        add(WARN, "/dev/shm", "探测失败（Windows 开发机正常）")
        return
    add(OK if mb >= 1024 else ERR, "/dev/shm",
        f"{mb:.0f}MB" + ("" if mb >= 1024 else
                         " —— 过小！docker run 加 --shm-size 2g"))


def check_deep():
    """torch_npu 实探：is_available 只看注册表，必须真分配一次。"""
    try:
        import torch
        import torch_npu  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        add(ERR, "torch_npu import", str(exc)[:120])
        return
    if not torch.npu.is_available():
        add(ERR, "torch.npu.is_available()", "False —— 拒绝静默降级 CPU，查卡/驱动")
        return
    add(OK, "torch.npu.is_available()", f"True, {torch.npu.device_count()} card(s)")
    try:
        dev = torch.device("npu:0")
        torch.zeros(1, device=dev)
        torch.npu.synchronize(dev)
        props = torch.npu.get_device_properties(0)
        add(OK, "npu:0 probe (zeros+sync)",
            f"{props.name} total={props.total_memory // 2**20}MB")
    except Exception as exc:  # noqa: BLE001
        add(ERR, "npu:0 probe (zeros+sync)", f"{str(exc)[:160]}（探卡失败——卡坏/上下文异常）")


def main():
    ap = argparse.ArgumentParser(description="NPU 环境预检")
    ap.add_argument("--deep", action="store_true",
                    help="附加 torch_npu 实探（需在含 torch_npu 的环境内）")
    args = ap.parse_args()
    check_devices()
    check_driver_mount()
    check_smi()
    check_shm()
    if args.deep:
        check_deep()
    n_err = sum(1 for lv, _, _ in results if lv == ERR)
    print(f"──── {len(results)} 项检查，{n_err} 个 ERROR ────")
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
