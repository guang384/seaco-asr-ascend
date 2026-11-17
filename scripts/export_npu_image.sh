#!/usr/bin/env bash
# NPU 镜像交付包导出（既定流程）：WSL ext4 内 save → gzip → gzip -t 完整性校验
# → 拷出目标目录 → 回收 WSL 侧临时文件。
# 绝不直接 docker save 到 /mnt/*（9p 链路有静默损坏风险）；产物名 = 镜像 tag 同源。
# 用法：bash export_npu_image.sh <镜像tag> [输出目录] [输出基名]
# 阶段标记：SAVE_OK / GZIP_OK / VERIFY_OK / COPY_OK / RECLAIM_OK / EXPORT_ALL_OK
set -euo pipefail

IMG="${1:?用法: export_npu_image.sh <镜像tag> [输出目录] [输出基名]}"
OUT_DIR="${2:-.}"
BASE="${3:-$(echo "$IMG" | tr ':/' '__')}"

F="$BASE.tar"
echo "STAGE_SAVE_BEGIN $(date '+%F %T')"
docker save "$IMG" -o "$F"
echo "SAVE_OK $(du -h "$F" | cut -f1)"
gzip -f "$F"
echo "GZIP_OK $(du -h "$F.gz" | cut -f1)"
gzip -t "$F.gz"
echo "VERIFY_OK"
mkdir -p "$OUT_DIR"
cp "$F.gz" "$OUT_DIR/"
echo "COPY_OK $(du -h "$OUT_DIR/$(basename "$F.gz")" | cut -f1)"
rm -f "$F.gz"
echo "RECLAIM_OK"
echo "EXPORT_ALL_OK $(date '+%F %T')  产物: $OUT_DIR/$(basename "$F.gz")"
