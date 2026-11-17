# ASR 推理服务 · 使用手册

supervisor 多进程流水线（supervisor + redis + 多 gateway + vad/punc/asr worker），配置全部走
**环境变量**（清单单一事实来源 = `proc_common.TUNABLE_PARAMS`，supervisor 启动
时逐项打印实际值）。目录结构与部署 quickstart 见 [README.md](README.md)。

| 端点 | 协议 | 场景 |
|---|---|---|
| `/v1/audio/transcriptions` | REST POST | 完整音频转写（吞吐优先），**双模式输入**：文件 / base64 |
| `/ws` | WebSocket | 实时流式转写，逐句返回（延迟优先） |

---

## 1. 启动与配置

容器启动（`deploy.sh run` 已封装）：

```bash
docker run -d --name asr --restart=unless-stopped --init --shm-size 2g \
  --device /dev/davinci0 --device /dev/davinci_manager \
  --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /root/.cache/ascend:/root/.cache/ascend \
  -e DEVICE=npu -e ASR_WORKERS=8 -e GATEWAY_WORKERS=2 -e MODE=mixed \
  -p 10095:10095 -p 10096:10096 iic-speech-seaco-paraformer-large-asr-ascend:arm-20261118
```

> 部署模型：**容器 = 单卡**。卡内全部 worker 吃同一个优先级队列（句子优先）；
> 跨卡扩容靠多容器（`deploy.sh scale N`，卡 i → 端口 10095+i）。

### 环境变量全表

| 变量 | 默认 | 说明 |
|---|---|---|
| `ASR_WORKERS` | 8 | **NPU 推理进程数**（容量主杠杆；64G 卡口径 ≈75% 水位，自检上限 10；32G 卡显式改 2-4。每 worker ≈ 权重 1.3GB + 0.3GB×批宽） |
| `ASR_MAX_BATCH_SIZE` | 16 | **同桶凑批宽**：一次认领 N 个"差不多长度"（±2s 桶）的 clip 一次批前向（预热/容量自检均按此宽对齐）；升级 funasr/torch 后跑 `scripts/validate_asr_batch.py` 验证批=逐条等价（funasr 必须与镜像同版） |
| `VAD_WORKERS` | 2 | VAD worker 进程数（CPU：整段切段 + 流式步进，独立队列） |
| `PUNC_WORKERS` | 2 | 标点 worker 进程数（CPU，独立队列） |
| `GATEWAY_WORKERS` | 2 | 网关进程数（SO_REUSEPORT 同端口分发；WS 会话粘在受理进程；无模型） |
| `MODE` | mixed | mixed=REST+WS 都服务；stream=纯流式（REST 503）；batch=纯批量（WS 拒连；VAD 照常加载——REST 切段必需） |
| `DEVICE` | ""（auto） | 强制设备 `npu`/`cpu`/`cuda`/`xpu`；auto=NPU→XPU→CUDA→CPU。**npu 探卡失败直接退出**（拒绝静默降 CPU——funasr 会静默回退，这正是要拦的），确需 CPU 显式 `DEVICE=cpu` |
| `ASCEND_DEVICE_ID` | "" | 容器内逻辑卡号（vNPU 池化下显式指定优先） |
| `MAX_PENDING_REQUESTS` | 8 | 每 gateway 在途 REST 请求闸门（超出 429） |
| `MAX_INFLIGHT_TASKS` | 512 | 任务队列容量（积压满 429 queue_full） |
| `WAIT_TIMEOUT_S` | 120 | 单请求总预算（超时 504 + 置取消标记，worker 跳过不白算） |
| `REDISPATCH_MAX` | 1 | **重试一次**：worker 死亡在途任务重派上限，超限弃置 → 请求 500 带原因 |
| `HEARTBEAT_STALE_S` / `STUCK_KILL_S` | 30 / 300 | 心跳停更判死 / busy 卡死强杀线（秒） |
| `SPAWN_READY_TIMEOUT_S` | 360 | NPU worker 串行启动的就绪等待上限（防并行 init 抢卡） |
| `WEDGE_RESTART_S` | 900 | 端到端假死判死（零成功且持续失败 → exit 交容器重启；0=禁用） |
| `SHM_ORPHAN_MAX_AGE_S` | 300 | 孤儿 SHM 块回收龄 |
| `MAX_AUDIO_MB` | 200 | 音频大小上限（解码前字节判断，413） |
| `IPC_REDIS_SOCKET` | /tmp/asr-redis.sock | 队列 unix socket（supervisor 自拉 redis-server） |
| `IPC_REDIS_URL` | "" | 外部 redis（`tcp://host:port`）：不拉起 server 直连（dev/多副本） |
| `METRICS_ENABLED` | 1 | `/metrics` 开关 |
| `PORT` / `ADMIN_PORT` | 10095 / 10096 | 业务 / 监控端口 |
| `LOG_LEVEL` | INFO | 日志级别 |
| `KEEP_ALIVE_TIMEOUT_S` | 0 | uvicorn keep-alive（0=逐请求 accept，长连接会粘死单 gateway） |

---

## 2. API 协议

### 2.1 REST `POST /v1/audio/transcriptions`（双模式输入）

流水为**边切边发**：VAD 按 30s 步进切段（与 WS 同一状态机、零边界误差），
每完成一段立即派发 ASR 识别——切段与识别重叠，长音频首句提前、总时长 ≈
max(切段, 识别)。重复段自动去重（worker 重派场景不重算）。

响应统一 `{"text": "…", "job_id": N}`（job_id 可用于 `/queue/{id}/promote` 插队）。

**模式 A：multipart/form-data**（OpenAI 兼容，`openai` SDK 可直连）

```bash
curl -F "file=@rec.wav" -F "hotword=魔搭,达摩院" \
  http://127.0.0.1:10095/v1/audio/transcriptions
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `file` | File（必填） | 音频 wav/mp3/flac/ogg（非 16k 自动重采样，多声道自动混音） |
| `hotword` | Text（可选） | 热词，逗号分隔 |
| `model` / `language` | Text（忽略） | 兼容 openai SDK |

**模式 B：application/json**（base64）

```bash
curl -H 'Content-Type: application/json' \
  -d "{\"audio_base64\": \"$(base64 -w0 rec.wav)\", \"hotword\": \"魔搭\"}" \
  http://127.0.0.1:10095/v1/audio/transcriptions
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `audio_base64` | String（必填） | base64 音频；支持 `data:audio/wav;base64,` 前缀自动剥离 |
| `hotword` | String（可选） | 热词，逗号分隔 |
| `filename` | String（可选） | 展示用文件名（进 /stats /queue 的任务名） |

**错误码**：

| 状态 | 场景 | 备注 |
|---|---|---|
| 200 | 成功 | `{"text","job_id"}`；无语音段 `{"text":"","job_id":0}`；空文件 `{"text":"","error":"empty file"}` |
| 400 | 坏 base64 / 缺字段 / 解码失败 | `detail` 含原因 |
| 413 | 音频超 `MAX_AUDIO_MB` | |
| 415 | Content-Type 不支持 | 用 multipart 或 JSON |
| 429 | 过载拒绝 | `Retry-After` + `X-Reject-Reason: queue_full \| shm_pressure` |
| 500 | 推理失败 | `detail` 含 worker 错误（含重试耗尽 "worker died twice"） |
| 503 | `MODE=stream` 无批量服务 | |
| 504 | 超 `WAIT_TIMEOUT_S` | 已置取消标记 |

### 2.2 WS `/ws`（流式句级）

- **上行**二进制帧，两种格式自动识别：
  - **wav 文件流**：首帧含 RIFF 头即解析（任意采样率/声道/位深自动转换），后续帧为数据；
  - **裸 PCM**：16kHz/16bit/mono（建议 200ms 一帧）。
- **结束**：文本帧 `{"is_final": true}`。
- **下行**逐句 JSON：`{"text": "…", "begin": 毫秒, "end": 毫秒, "is_final": false}`，
  流结束补 `{"is_final": true}`；格式不支持 `{"error": "…"}`。
- mp3 等压缩格式不走 WS（只走 REST）；`MODE=batch` 拒连（close 1013）。

---

## 3. 观测端点

**业务端口 10095**：

| 端点 | 说明 |
|---|---|
| `GET /health` | 五键健康卡 `{status, ready, version, shm, recent_failures}`；ready=false → 503（容器 HEALTHCHECK / K8s 三探针共用） |
| `GET /metrics` | Prometheus 文本：`asr_e2e_*`（requests/successes/timeouts/rejections/failed/sentences）、`asr_queue_*`（depth/in/out/bad）、**`asr_queue_wait_ms{queue="vad\|asr\|punc",quantile="avg\|p50\|p95\|max"}` 各队列任务等待时间**、`asr_worker_*{role}`（ready/stuck_seconds/jobs/errors/crashes/npu_mem_allocated_mb；asr 另有**凑批观测** `batch_claims_total`/`batch_tasks_total`/`batch_max`/`batch_size_total{bucket}`）、`asr_shm_*`、`asr_degraded`、`asr_build_info` |
| `GET /docs` | Swagger |

**admin 端口 10096**：

| 端点 | 说明 |
|---|---|
| `GET /ready` | 就绪探针（200/503，与 /health 同源，附 expected/ready_counts 明细） |
| `GET /stats` | 累计统计：processed/failed/requests/timeouts/sentences/音频总时长/墙钟/RTF/最近 10 任务（跨 gateway 心跳聚合，**≤5s 延迟**） |
| `GET /queue` | **三队列状态（深度/吞吐/等待时间 avg/p50/p95/max）** + 各 gateway 在途 + 容量估算 |
| `POST /queue/{job_id}/promote` | 任务插队（clip 优先级 100→10，仍让位句子；不在队列 404） |
| `GET /resources` | CPU/内存 + npu-smi + /dev/shm 水位 |

容量估算口径（`/queue` 的 `capacity`）：
`ws_capacity = min(VAD 每 worker 60 路 × VAD_WORKERS, ASR_WORKERS / (单句占用秒 × 0.2句/s/路))`；
`batch_capacity = ASR_WORKERS × 平均音频秒/平均推理秒`。先验（句 0.5s / 15x）起步，
滚动样本 ≥10 条后自动校准（`calibrated: true`）。

---

## 4. 故障语义（自愈链）

| 故障 | 行为 |
|---|---|
| worker 前向 OOM | `empty_cache` + **整批重试一次**；仍败降级逐条隔离，坏条 → 该任务 500 |
| worker 进程死亡 | supervisor 秒级**精确重派**在途任务（换 worker 重算，`REDISPATCH_MAX=1`）；二次死亡 → 弃置 + 500 带原因（"worker died twice"） |
| worker 推理卡死（aicore 挂起不抛错） | BusyWatch `stuck_s > STUCK_KILL_S(300s)` → supervisor 强杀重生（指数退避 `min(30, 0.5·2^n)`，稳定 60s 清零） |
| 心跳停更 ≥30s | 判死强杀重生 |
| redis-server 死亡 | supervisor 自动重启；队列清零，在途请求走 504 兜底 |
| 请求超时（>WAIT_TIMEOUT_S） | 504 + 取消标记（worker 认领后跳过不白算） |
| 端到端假死（全角色就绪但零成功且持续失败） | supervisor `exit(1)` 交容器重启（`WEDGE_RESTART_S=900`） |

---

## 5. 测试

```bash
# 离线语义回归（Windows/Linux 均可，fakeredis + 真 SHM，不加载模型）
python -m pytest -q

# 容器内全量（镜像自带 tests/，fakeredis 语义回归）
docker run --rm --entrypoint python3 iic-speech-seaco-paraformer-large-asr-ascend:arm-20261118 -m pytest -q

# 故障注入（对着已起的服务：杀 worker 验证重派 + 自愈闭环）
docker exec asr python3 scripts/parity_smoke.py --audio /app/test.wav

# NPU 预检（部署前）：stdlib 档 + 镜像内 --deep 真探
./deploy.sh preflight
```

覆盖语义：优先级序（句子<插队<clip）、认领/ack 原子性、死亡重派（重试一次
再报错）、毒块弃置、promote、取消标记、SHM 读写/幂等清理、双模式输入路由
（multipart/base64/data URI/坏 base64 400/超限 413/超时 504/空文件）、WS 流式
（句级 JSON + is_final + batch 拒连）、心跳 stale/stuck 判决、/health 五键卡。

---

## 6. 压测与调参

```bash
# 压测语料：AISHELL-4 中文会议长录音截取的 50 个不同长度区间（bench_audio/，
# MANIFEST.md 溯源；重建 python scripts/build_bench_corpus.py 50；
# 用你自己的录音不出机：python scripts/build_bench_corpus.py --cut-from 目录/）

# REST 并发压测（file / base64 两种输入各一轮，对照链路等价性）
python run_batch_http.py bench_audio out.txt 4 file
python run_batch_http.py bench_audio out-b64.txt 4 base64
# 输出：墙钟总耗时、吞吐倍速、单请求 p50/p90/max → 输出文件
#（服务端基准看 /metrics 的 asr_queue_wait_ms——各队列等待时间是过载的第一信号）

# 随机抽样压测（持续施压：目录内随机抽音频，测饱和容量/时延分布/稳定性）
python scripts/stress_asr.py bench_audio -c 4 -d 60            # 4 并发压 60s
python scripts/stress_asr.py bench_audio -c 8 -n 200 --rps 2   # 定速 2 请求/秒共 200 发
# 输出：控制台汇总（时延 p50/p90/p95/max、音频倍速、HTTP/异常分布）+ 逐请求明细 CSV
# 与 run_batch_http.py 的分工：那个每文件恰好一次（单轮吞吐）；本工具随机抽样持续施压

# 凑批质量判据（配比 8×16 / 10×12 / 16×8 决策用，压测时看 /metrics）：
#   平均批宽 = asr_worker_batch_tasks_total / asr_worker_batch_claims_total
#   凑批率   = 平均批宽 ÷ ASR_MAX_BATCH_SIZE
#   长期 <50% → 供给不足，减 worker 或减批宽；长期 ≈100% → 批宽还有上探空间

# 离线批量吞吐（不经 HTTP；batch_size_s 扫描）
python asr_batch.py --input bench_audio --batch-size-s 30
```

调参决策表：

| 症状 | 动作 |
|---|---|
| 流式句级延迟高、容量不够 | 加 `ASR_WORKERS`（每 worker ≈ 1.3GB + 0.3GB×批宽；64G 卡 8 起步，32G 卡 2-4；启动有容量自检） |
| 批量吞吐低 | 调 `ASR_MAX_BATCH_SIZE`（默认 16，同桶凑批；用压测扫 8/16/32 找拐点）；或加 worker |
| 429 queue_full | 加 worker / gateway，或调 `MAX_PENDING_REQUESTS` / `MAX_INFLIGHT_TASKS` |
| 429 shm_pressure | 加 `--shm-size`；降并发 |
| WS 会话多但 VAD 串行 | 加 `VAD_WORKERS`（每 worker ~60 路流式） |
| 跨卡线性扩展 | `deploy.sh scale N`（N 个单卡容器） |

**910B 全部基线数字待实测**（README 的 iGPU 数字仅作数量级参考）。

---

## 7. WSL 构建 + 导出（本地交叉构建交付）

**验证矩阵**（QEMU 仿真跑不了 torch 推理——xbyak_aarch64 CPU 探测段错误，kokoro 同款已知坑）：

| 验证项 | 跑在哪 | 说明 |
|---|---|---|
| 离线语义测试（66 用例） | 本地 Windows / arm64 容器内 | stub + fakeredis，不碰 torch，QEMU 可跑 |
| 真模型全拓扑冒烟 | **原生 x86 CPU 镜像**（WSL）或 910B `deploy.sh smoke` | QEMU 会段错误，别硬跑 |
| NPU 推理冒烟 | 910B 服务器 `deploy.sh smoke` | 必须真卡 |

```bash
wsl --shutdown && sleep 8        # 构建前清理（vhdx 只涨不缩，见全局备忘）
wsl -d Ubuntu-22.04 -u root

# ① arm64 NPU 镜像（交付物）
docker build --platform linux/arm64 -f Dockerfile_NPU -t iic-speech-seaco-paraformer-large-asr-ascend:arm-20261118 \
  /mnt/c/Users/<你的用户名>/Desktop/ASR
docker run --rm --platform linux/arm64 --ipc=host --entrypoint python3 \
  iic-speech-seaco-paraformer-large-asr-ascend:arm-20261118 -m pytest -q          # 容器内 66 用例（stub，QEMU 可跑）

# ② 真模型全拓扑冒烟：原生 x86 CPU 镜像（无仿真，真模型+supervisor+双模式+WS）
docker build -f Dockerfile_CPU -t asr-cpu:local /mnt/c/Users/<你的用户名>/Desktop/ASR
docker run -d --name asr-smoke --shm-size 2g -e DEVICE=cpu -e ASR_WORKERS=1 \
  -e VAD_WORKERS=1 -e PUNC_WORKERS=1 -e GATEWAY_WORKERS=1 -p 10095:10095 asr-cpu:local
#   等 /health ready=true 后按 §2 手工 curl（文件+base64+WS 三链路），完事 docker rm -f asr-smoke

# ③ 导出（WSL ext4 内 save → 校验 → 拷出 → 回收）
docker save iic-speech-seaco-paraformer-large-asr-ascend:arm-20261118 \
  -o ~/iic-speech-seaco-paraformer-large-asr-ascend-arm-20261118.tar \
  && gzip -f ~/iic-speech-seaco-paraformer-large-asr-ascend-arm-20261118.tar
gzip -t ~/iic-speech-seaco-paraformer-large-asr-ascend-arm-20261118.tar.gz \
  && cp ~/iic-speech-seaco-paraformer-large-asr-ascend-arm-20261118.tar.gz /mnt/c/Users/<你的用户名>/Desktop/ \
  && rm -f ~/iic-speech-seaco-paraformer-large-asr-ascend-arm-20261118.tar.gz
```

---

## 8. FAQ

**Q: 提示 "torch_npu 已加载但没有任何可用 NPU 卡"？**
设备解析 fail-fast（拒绝静默降 CPU 伪装健康）。查 `npu-smi info`、容器
`--device` 挂载、`/usr/local/Ascend/driver`、`ASCEND_DEVICE_ID`；
跑 `./deploy.sh preflight`。开发机确需 CPU：显式 `DEVICE=cpu`。

**Q: 首个请求要几十秒甚至几分钟？**
NPU 冷 JIT 编译。`/root/.cache/ascend` 挂载后只付一次（deploy.sh 已挂）；
worker 启动时已有 0.5s+30s 假音频热身，正常情况请求不该再遇编译。

**Q: 容器反复重启、日志报 SIGBUS？**
`/dev/shm` 不足（音频 SHM 写满 tmpfs）。`docker run --shm-size 2g`；
K8s 挂 `emptyDir: {medium: Memory}`（asr-deployment.yaml 已带）。

**Q: /health 一直 503？**
`ready` 附带 `expected`/`ready_counts` 明细：哪个角色缺员一目了然。
常见：asr worker 加载慢（启动期正常）、NPU 探卡失败（worker 反复退出，
看 `docker logs asr` 的 respawn 行与 fail-fast 报错）。

**Q: 请求 500 "task dropped: worker died twice"？**
同一任务的两次 worker 都死了（重试一次语义的上限）。属极端故障，
`/metrics` 看 `asr_worker_crashes_total` 定位崩因（多为 OOM/卡坏态）。

**Q: base64 和文件两种输入结果不一样？**
不应该。两条链路共享同一解码→VAD→队列→worker 管线，仅入口解析不同；
用 `run_batch_http.py` 的 file/base64 两轮压测对照（§6）。

**Q: 怎么把某个慢任务提前？**
`curl -X POST http://127.0.0.1:10096/queue/{job_id}/promote`（job_id 在 REST
响应里）。只影响还在队列里的 clip（100→10），跑着的不动。

**Q: docker exec 进去跑 python 报 CANN 环境缺失？**
`bash scripts/with_ascend_env.sh python3 …`（重入 CANN 环境）。

**Q: 升级 funasr 版本注意什么？**
当前钉 `funasr==1.4.21 + kaldi-native-fbank`（kokoro 生产验证组合，无
torchaudio）。升级前必须重跑容器内 pytest + CPU 冒烟对照识别文本。
