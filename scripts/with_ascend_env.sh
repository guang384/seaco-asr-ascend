#!/usr/bin/env bash
# 复刻镜像 ENTRYPOINT 的 CANN 环境，供 docker exec 等不经过 ENTRYPOINT 的命令使用。
# 用法：bash scripts/with_ascend_env.sh python3 scripts/parity_smoke.py
# 移植自 kokoro-tts-ascend/scripts/with_ascend_env.sh。
set -e
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi="${ATB_CXX_ABI:-1}"
fi
exec "$@"
