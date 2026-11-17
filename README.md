# ASR 推理服务（昇腾 910B / supervisor 多进程流水线）

中文语音识别（SeACo-Paraformer）推理服务，**supervisor 多进程流水线**：
supervisor 编排 redis 队列 + 多 gateway（SO_REUSEPORT）+ **vad / punc / asr
三类独立 worker 进程池（各自队列）**，优先级调度 + 故障自愈 + 各队列等待时间进指标。

- **REST** `POST /v1/audio/transcriptions`（OpenAI 兼容）：完整音频转写，
  **双模式输入**——multipart 文件上传 / JSON `audio_base64`
- **WS** `/ws`：实时流式，逐句返回，延迟优先
- 三条队列同构优先级：**流式/句尾（0）永远先于批量（100，插队后 10）**被认领
- **故障自愈**：worker 挂死 → supervisor 秒级重派在途任务（重试一次，再失败
  客户端立即 500 带原因）→ 看门狗超时强杀重生（NPU aicore 挂起、推理卡死全覆盖）

使用手册见 [USAGE.md](USAGE.md)（环境变量、API 协议、观测端点、测试、调参压测、FAQ）。

## 进程拓扑

```
supervisor (PID 1, 绝不 import torch_npu)
├── redis-server         unix socket /tmp/asr-redis.sock（纯内存、有界）
├── gateway ×E           SO_REUSEPORT 同端口内核哈希分发（无模型）
│     REST 双模式输入 + WS 流式会话 + 音频解码
│     SHM 音频块 → 三级流水编排；结果邮箱聚合 → 响应
│     /health 五键卡 + /metrics（含队列等待时间）+ admin 端口
├── vad-worker ×V   [vad_queue]   FSMN-VAD（CPU）：整段切段 / 流式步进(cache 往返)
├── punc-worker ×P  [punc_queue]  CT-Transformer（CPU）：句尾 / 全文标点
└── asr-worker ×N   [asr_queue]   SeACo-Paraformer（NPU fail-fast 探卡）：句子 / clip
      三队列同构 ZSET 优先级认领（流式 0 < 插队 10 < 批量 100）
      心跳文件 + BusyWatch 看门狗；asr OOM 自愈（empty_cache + 重试一次）
      每任务推送等待时间样本 → /metrics asr_queue_wait_ms{queue,quantile}
```

REST 流水（**边切边发**）：解码 → `[vad 整段切段]`（30s 步进，增量段随到随
派发 clip）→ `[asr 并行识别]`（与切段重叠）→ `[punc 全文标点]`——首段不等整段
VAD，长音频总时长 ≈ max(切段, 识别) 而非相加；
WS 流水：帧缓冲 → `[vad 流式步进]` → 句尾 `[asr 句子]` → `[punc 句尾标点]`。

任务失败语义：worker 死亡 → 在途任务**重试一次**（换 worker 重算）→ 仍失败
弃置并补错误回网关（`500` 带原因，不挂到超时）。

## 目录结构

```
supervisor.py          # PID 1：进程编排 + 看门狗 + 死亡重派（绝不 import torch_npu）
gateway.py             # 网关进程（无模型）：REST/WS 接入 + 解码 + 三级流水编排 + 观测
compute.py             # worker 进程：vad / punc / asr 三角色（各自队列认领消费）
proc_common.py         # IPC 基座：Redis 队列（认领式/邮箱）+ 心跳 + BusyWatch + 等待采样 + env
protocol.py            # 跨进程消息契约（vad/asr/punc 三对 task/result）
asr_common.py          # 设备探测（fail-fast）+ 模型加载
asr_infer.py           # 单文件推理 CLI（冒烟用）
asr_batch.py           # 批量离线转写 CLI（batch_size_s 扫描用）
run_batch_http.py      # REST 压测客户端（file/base64 双模式）
deploy.sh              # 910B 部署：preflight / build / run / scale / smoke
scripts/               # npu_probe_check.py 预检 · parity_smoke.py 故障注入
                       # with_ascend_env.sh · build_bench_corpus.py 压测语料构建
tests/                 # 离线语义回归（fakeredis + 真 SHM，Windows 可跑）
Dockerfile_NPU         # 昇腾 910B 镜像（ascendhub torch-npu + CANN，arm64）
Dockerfile_CPU         # 纯 CPU 镜像（x86_64 / arm64）
asr-deployment.yaml    # K8s（startup/readiness/liveness 共用 /health）
requirements.txt       # Python 依赖（funasr+kaldi-native-fbank，无 torchaudio）
speech_seaco_paraformer_.../speech_fsmn_vad_.../punc_ct-transformer_.../  # 三模型
bench_audio/           # 压测语料（AISHELL-4 中文会议截取，50 个不同长度，MANIFEST 溯源）
录音样本/               # 压测数据（涉密，不入库不进镜像）
test_two_sentences.wav # 冒烟测试音频（涉密合规：**任何音频不打进镜像**，smoke/压测音频运行期宿主挂载）
```

## 模型

三个 ModelScope 模型，**目录名即模型 id**（[asr_common.py](asr_common.py) 按相对路径加载，
禁用在线更新；模型 ~1.2GB 直接打进镜像）：

| 目录（= ModelScope id）                                | 作用             | 部署位置      |
| --------------------------------------------------- | -------------- | ---------- |
| `speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch` | ASR 主模型（含热词） | NPU（每 worker 一份） |
| `speech_fsmn_vad_zh-cn-16k-common-pytorch`           | VAD 切句         | CPU（独立 vad-worker 进程池：整段切段 + 流式步进） |
| `punc_ct-transformer_zh-cn-common-vocab272727-pytorch` | 标点             | CPU（独立 punc-worker 进程池：句尾/全文） |

丢失后重新下载：

```bash
pip install modelscope
modelscope download --model speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
modelscope download --model speech_fsmn_vad_zh-cn-16k-common-pytorch
modelscope download --model punc_ct-transformer_zh-cn-common-vocab272727-pytorch
```

## 910B 部署 quickstart

```bash
# 0. 提取部署脚本（随镜像分发，版本永不漂移）；镜像 tar 交付时先 docker load
docker run --rm <镜像tag> cat /app/deploy.sh > deploy.sh && chmod +x deploy.sh

# 1. 部署前预检（设备/驱动/shm；镜像已 load 时附 torch_npu 真探）
./deploy.sh preflight

# 2. 服务器上构建（或本地 WSL 构建后 load tar.gz，见下）
./deploy.sh build

# 3. 起服务（单卡 8 worker = 64G 卡口径 × 2 gateway；32G 卡 run 末尾传 2-4）
./deploy.sh run 0 10095 10096 8 mixed

# 4. 三级冒烟：①单文件 NPU 推理 ②双模式 curl ③WS 流式
./deploy.sh smoke 0

# 5. 故障注入验证重派语义（对着已起的服务）
docker exec asr python3 scripts/parity_smoke.py --audio /app/test.wav

# 6. 压测定 worker 数（file/base64 两种输入各压一轮对照）
python run_batch_http.py 录音样本 out.txt 4 file
python run_batch_http.py 录音样本 out-b64.txt 4 base64
python scripts/stress_asr.py bench_audio -c 4 -d 60   # 持续随机抽样压测（时延分布/稳定性）
```

**本地 WSL 构建 arm64 镜像**（服务器无外网时的交付路径；验证矩阵见 USAGE §7）：

```bash
wsl -d Ubuntu-22.04 -u root
docker build --platform linux/arm64 -f Dockerfile_NPU -t iic-speech-seaco-paraformer-large-asr-ascend:arm-20261118 /mnt/c/Users/<你的用户名>/Desktop/ASR
docker run --rm --platform linux/arm64 --ipc=host --entrypoint python3 \
  iic-speech-seaco-paraformer-large-asr-ascend:arm-20261118 -m pytest -q        # 容器内 66 用例（stub 测试，QEMU 可跑）
# 真模型全拓扑冒烟用原生 x86 CPU 镜像（QEMU 跑 torch 推理会段错误）：
docker build -f Dockerfile_CPU -t asr-cpu:local /mnt/c/Users/<你的用户名>/Desktop/ASR
# ……起容器 curl 文件+base64+WS，见 USAGE §7
docker save iic-speech-seaco-paraformer-large-asr-ascend:arm-20261118 \
  -o ~/iic-speech-seaco-paraformer-large-asr-ascend-arm-20261118.tar \
  && gzip -f ~/iic-speech-seaco-paraformer-large-asr-ascend-arm-20261118.tar \
  && gzip -t ~/iic-speech-seaco-paraformer-large-asr-ascend-arm-20261118.tar.gz
cp ~/iic-speech-seaco-paraformer-large-asr-ascend-arm-20261118.tar.gz /mnt/c/Users/<你的用户名>/Desktop/ \
  && rm -f ~/iic-speech-seaco-paraformer-large-asr-ascend-arm-20261118.tar.gz
```

NPU 容器**必须**有 `/dev/shm ≥ 2g`（音频 SHM 块写满 64MB 默认 tmpfs 即 SIGBUS）
与 `/root/.cache/ascend` 挂载（CANN 算子 JIT 缓存持久化，否则每次重启重编译）——
`deploy.sh run` 均已包含；K8s 见 `asr-deployment.yaml` 的 `dshm` volume。

**K8s**：[asr-deployment.yaml](asr-deployment.yaml)，三探针共用业务端口 `/health`。

## 性能基线

**iGPU 参考基线（Arc 130T / 2 实例 / 2026-09，旧单进程版实测）**：

| 场景          | 结果                                          |
| ----------- | ------------------------------------------- |
| 流式 8 路      | 句级 p50 1.8s / p90 3.3s（含尾静音 800ms）          |
| 流式容量上限     | ~20 路（吞吐 3.1 句/s 封顶）                        |
| 批量顺序 / 4 路并发 | 26.8x / 28.7x 实时                             |
| 混合 8 路 + 批量 | 流式 p90 与纯流式持平；批量 16.5x（空闲填隙）                |

**910B 数字待实测**（换硬件后全部重测，方法见 USAGE §6）：
单句预计 60~120ms 级；线上实况看 `/queue` 的 `ws_capacity` / `batch_capacity`
（先验起步，流量后自动校准）。

## 后续优化线索

1. ~~clip 张量组批~~ **已实现（同桶凑批）**：worker 一次认领 N 个"差不多长度"
   （±2s 时长桶）的 clip，一次 `generate` 批前向（`ASR_MAX_BATCH_SIZE`，默认 16）。
   数值等价门：`scripts/validate_asr_batch.py`（本地实测批=逐条文本全一致，2.5-3.1x）。
2. **910B 重扫 `ASR_MAX_BATCH_SIZE`**：默认已提到 16（64G 卡口径）——用
   `scripts/stress_asr.py` 扫 8/16/32 找吞吐拐点（更大批受同桶供给限制）。
3. **显存 → worker 数**：每 worker ≈ 权重 1.3GB + 0.3GB×批宽（16 档 ≈ 6.0GB）。
   64G 卡默认 8（≈75% 水位，自检口径上限 10）；32G 卡显式调低 2-4。
   `supervisor.py _capacity_check` 启动自检（批感知公式）；显存为动态分配
   （不预占，随批形状伸缩；共享卡/vNPU 隔离场景如需硬限再议）。
4. **换硬件重测基线**：方法与决策表见 USAGE.md §6。

## 已知坑（踩过的，别再踩）

- funasr **AutoModel 线程不安全** → 进程化后结构性消解（每 worker 进程独占）
- iGPU 上 `torch.autocast` fp16 → ASR **静默输出空文本**（不报错），别开
- funasr 在 NPU 不可用时**静默回退 CPU**（把卡坏了伪装成健康）——`asr_common.
  resolve_device` 显式 `DEVICE=npu` 时 fail-fast 拦截；`auto` 档逐级降级但
  **每级失败打 WARNING**（降级在日志里看得见，绝不无声）
- **心跳线程绝不碰 torch.npu**（设备坏态副线程碰 NPU 会 SIGBUS 杀进程，
  kokoro 生产事故 20261109）——占用数据由主循环采样写缓存
- NPU 首启冷 JIT 编译分钟级：`/root/.cache/ascend` 挂载后只付一次；
  HEALTHCHECK/startupProbe 宽限已覆盖（360s/15min）
- `/dev/shm` 默认 64MB：音频 SHM 写满即 SIGBUS，`--shm-size 2g` 必带
- VAD 在加速卡上会与 ASR forward 互相卡（队头阻塞），所以 VAD 全在 CPU
- Windows 终端 PYTHONPATH 污染会炸 numpy（`$env:PYTHONPATH=$null` 后再跑）

## 许可

- 本仓库代码：**Apache License 2.0**（见 [LICENSE](LICENSE)）
- 推理模型（权重不入库；镜像内权重随 ModelScope 原许可分发）：
  [iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch](https://modelscope.cn/models/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch)、
  [iic/speech_fsmn_vad_zh-cn-16k-common-pytorch](https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch)、
  [iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch](https://modelscope.cn/models/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch)
  ——三者模型 README 自带元数据均为 **Apache License 2.0**
- `bench_audio/` 压测集截取自 [AISHELL-4](https://www.aishell.net/)（**CC BY 4.0**），
  溯源见 [bench_audio/MANIFEST.md](bench_audio/MANIFEST.md)
- `录音样本/` 为私有电话录音，**不入库、不分发**
