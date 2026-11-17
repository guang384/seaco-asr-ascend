#!/bin/bash
# 910B 服务器部署脚本：预检 + 构建镜像 + 起容器 + 冒烟
# 本脚本已烧进镜像（/app/deploy.sh），服务器上一行提取（版本随镜像走）：
#   docker run --rm <镜像tag> cat /app/deploy.sh > deploy.sh && chmod +x deploy.sh
# 用法：
#   ./deploy.sh preflight            # 部署前预检（设备/驱动/shm；镜像已 load 时附 torch_npu 实探）
#   ./deploy.sh build [BASE_IMAGE]   # 构建镜像
#   ./deploy.sh run [NPU_ID] [业务端口] [监控端口] [worker数] [模式] [gateway数]
#                                    # 单卡起容器（默认 0 / 10095 / 10096 / 4 / mixed / 2）
#   ./deploy.sh scale [容器数] [基础业务端口] [worker数] [模式]
#                                    # N 个单卡容器（卡 i → 端口 基础+i；默认 8 / 10095 / 4 / mixed）
#   ./deploy.sh smoke [NPU_ID] [音频文件]
#                                    # 三级冒烟：①单文件 NPU 推理 ②拓扑双模式 curl ③WS 流式一连
#   （故障注入重派验证，对着已起的服务跑：docker exec asr python3 scripts/parity_smoke.py）
set -e
IMAGE=${ASR_IMAGE:-iic-speech-seaco-paraformer-large-asr-ascend:arm-20261118}   # 环境变量可覆盖：ASR_IMAGE=xxx ./deploy.sh ...

cmd=${1:-run}
BASE=${2:-swr.cn-south-1.myhuaweicloud.com/ascendhub/torch-npu:2.14.0-cann9.2.0-910b-ubuntu22.04-py3.12}
NPU_ID=${2:-0}
PORT=${3:-10095}
ADMIN_PORT=${4:-10096}

# NPU 容器公共挂载：设备 + 驱动 + 编译缓存（缓存持久化后算子 JIT 编译只付一次代价）
npu_mounts() {
  echo "--device /dev/davinci${NPU_ID} --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc -v /usr/local/Ascend/driver:/usr/local/Ascend/driver -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi -v /etc/ascend_install.info:/etc/ascend_install.info -v /root/.cache/ascend:/root/.cache/ascend"
}

# 生产容器公共参数：崩溃自愈 + 僵尸回收 + 日志上限 + SHM 容量
# --shm-size 2g 必带：音频块写满默认 64MB tmpfs 即 SIGBUS（kokoro 生产事故）
# 健康检查打业务端口 /health（start-period 覆盖加载 + 串行 JIT 热身）
run_flags() {
  echo "--restart=unless-stopped --init --shm-size 2g \
--log-opt max-size=10m --log-opt max-file=3 \
--health-cmd 'curl -sf http://127.0.0.1:10095/health || exit 1' \
--health-interval=30s --health-timeout=5s --health-start-period=360s --health-retries=3"
}

need_image() {
  if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "镜像 $IMAGE 不存在：先 ./deploy.sh build，或 docker load -i <tar.gz>，" >&2
    echo "或 ASR_IMAGE=<tag> 指定其他 tag" >&2
    exit 1
  fi
}

case "$cmd" in
preflight)
  echo "── 部署前预检 ──"
  ok=1
  for dev in /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc; do
    [ -e "$dev" ] && echo "[OK ] $dev" || { echo "[ERR] $dev 缺失（容器需 --device 挂载）"; ok=0; }
  done
  ls /dev/davinci[0-9]* >/dev/null 2>&1 && echo "[OK ] davinci 节点: $(ls /dev/davinci[0-9]* | tr '\n' ' ')" \
    || { echo "[ERR] 无 /dev/davinciN 节点"; ok=0; }
  [ -d /usr/local/Ascend/driver ] && echo "[OK ] /usr/local/Ascend/driver 已挂载" \
    || { echo "[ERR] /usr/local/Ascend/driver 缺失"; ok=0; }
  if command -v npu-smi >/dev/null 2>&1; then
    npu-smi info >/dev/null 2>&1 && echo "[OK ] npu-smi info 正常" || { echo "[ERR] npu-smi info 失败（驱动未就绪？）"; ok=0; }
  else
    echo "[WARN] npu-smi 不在 PATH（仅影响预检展示）"
  fi
  shm_mb=$(df -m /dev/shm 2>/dev/null | awk 'NR==2{print $2}')
  if [ -n "$shm_mb" ]; then
    [ "$shm_mb" -ge 1024 ] && echo "[OK ] /dev/shm ${shm_mb}MB" \
      || { echo "[ERR] /dev/shm 仅 ${shm_mb}MB —— run 时必须 --shm-size 2g"; ok=0; }
  fi
  if docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "── 镜像内 torch_npu 实探（--deep）──"
    docker run --rm $(npu_mounts) --entrypoint python3 "$IMAGE" \
      scripts/npu_probe_check.py --deep || ok=0
  else
    echo "[SKIP] 镜像未 load，跳过 torch_npu 实探（build/load 后重跑 preflight）"
  fi
  [ "$ok" = 1 ] && echo "预检通过" || { echo "预检存在问题（见 [ERR]）"; exit 1; }
  ;;
build)
  docker build -f Dockerfile_NPU -t "$IMAGE" --build-arg BASE_IMAGE="$BASE" .
  echo "构建完成: $IMAGE"
  ;;
run)
  need_image
  N_INST=${5:-8}     # 64G 卡默认（batch16 档 ~6.0GB/worker）；32G 卡请显式传 2-4
  MODE_ARG=${6:-mixed}
  N_GW=${7:-2}
  docker rm -f asr 2>/dev/null || true
  # shellcheck disable=SC2046
  docker run -d --name asr \
    $(npu_mounts) $(run_flags) \
    -e DEVICE=npu -e ASCEND_DEVICE_ID=${NPU_ID} \
    -e ASR_WORKERS=${N_INST} -e GATEWAY_WORKERS=${N_GW} -e MODE=${MODE_ARG} \
    -p ${PORT}:10095 -p ${ADMIN_PORT}:10096 \
    "$IMAGE"
  echo "业务: http://$(hostname -I | awk '{print $1}'):${PORT}  监控: :${ADMIN_PORT}  卡${NPU_ID} | ${N_INST} worker × ${N_GW} gateway · ${MODE_ARG}  日志: docker logs -f asr"
  ;;
scale)
  need_image
  N_CONT=${2:-8}
  BASE_PORT=${3:-10095}
  N_INST=${4:-8}     # 每容器（单卡）worker 数；64G 卡默认，32G 卡传 2-4
  MODE_ARG=${5:-mixed}
  for i in $(seq 0 $((N_CONT-1))); do
    # shellcheck disable=SC2046
    docker run -d --name asr-$i \
      $(NPU_ID=$i npu_mounts) $(run_flags) \
      -e DEVICE=npu -e ASCEND_DEVICE_ID=$i \
      -e ASR_WORKERS=${N_INST} -e GATEWAY_WORKERS=2 -e MODE=${MODE_ARG} \
      -p $((BASE_PORT+i)):10095 -p $((BASE_PORT+1000+i)):10096 \
      "$IMAGE"
    echo "容器 asr-$i → 卡 $i，业务 :$((BASE_PORT+i)) 监控 :$((BASE_PORT+1000+i))"
  done
  echo "共 ${N_CONT} 个单卡容器。扩缩容：docker rm -f asr-i；日志：docker logs -f asr-i"
  ;;
smoke)
  need_image
  AUDIO=${3:-test_two_sentences.wav}
  echo "── ① 单文件 NPU 推理冒烟 ──"
  docker run --rm --entrypoint python3 \
    $(npu_mounts) \
    -v "$(pwd)/${AUDIO}:/app/test.wav:ro" \
    "$IMAGE" asr_infer.py --device npu --audio /app/test.wav
  echo "── ② 拓扑冒烟（双模式输入）──"
  docker rm -f asr-smoke 2>/dev/null || true
  # shellcheck disable=SC2046
  docker run -d --name asr-smoke $(npu_mounts) $(run_flags) \
    -e DEVICE=npu -e ASCEND_DEVICE_ID=${NPU_ID} -e ASR_WORKERS=1 -e GATEWAY_WORKERS=1 \
    -p 11095:10095 -p 11096:10096 \
    "$IMAGE" >/dev/null
  echo "等待 /health ready（加载 + JIT 热身，最长 6 分钟）..."
  for _ in $(seq 1 72); do
    if curl -sf http://127.0.0.1:11095/health >/dev/null 2>&1; then break; fi
    sleep 5
  done
  curl -sf http://127.0.0.1:11095/health >/dev/null || { echo "smoke 容器未就绪"; docker logs asr-smoke | tail -20; docker rm -f asr-smoke; exit 1; }
  echo "--- multipart 文件 ---"
  T1=$(curl -sf -F "file=@${AUDIO}" http://127.0.0.1:11095/v1/audio/transcriptions)
  echo "$T1"
  echo "--- JSON base64 ---"
  # base64 载荷超 Linux 单 exec 参数上限（128KB）——写文件后 curl @file
  printf '{"audio_base64": "%s"}' "$(base64 -w0 "$AUDIO")" > /tmp/asr_req.json
  T2=$(curl -sf -H 'Content-Type: application/json' \
    -d @/tmp/asr_req.json \
    http://127.0.0.1:11095/v1/audio/transcriptions)
  echo "$T2"
  echo "── ③ WS 流式一连（python3 客户端）──"
  docker exec asr-smoke python3 - <<'EOF'
import asyncio, json, wave, websockets
async def main():
    async with websockets.connect("ws://127.0.0.1:10095/ws") as ws:
        with wave.open("/app/test.wav", "rb") as w:
            data = w.readframes(w.getnframes())
        for i in range(0, len(data), 6400):
            await ws.send(data[i:i+6400])
        await ws.send(json.dumps({"is_final": True}))
        while True:
            m = json.loads(await ws.recv())
            print("WS <-", m)
            if m.get("is_final"):
                break
asyncio.run(main())
EOF
  docker rm -f asr-smoke >/dev/null
  echo "冒烟通过（①NPU 推理 ②双模式 text 应一致 ③WS 逐句输出）"
  ;;
*)
  echo "用法: $0 {preflight|build|run|scale|smoke}"; exit 1
  ;;
esac
