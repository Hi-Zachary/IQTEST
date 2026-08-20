#!/bin/bash
# 串行训练脚本（睡觉自动跑）：
#   1) AGIQA-3K 消融 A0/A1/A2/A3（每个 100ep，同一协议）
#   2) AGIQA-3K A3 = 完整模型（复用上面的 A3 运行，不重复）
#   3) AIGCIQA2023 A3 = 第二数据集完整模型（100ep）
# 每个任务跑完等显卡空闲后再开始下一个；任一失败即退出。
set -uo pipefail

PY=/root/autodl-tmp/CondaEnv/ipiqa/bin/python
ROOT=/root/autodl-tmp/newIdea_extracted/newIdea/ipqa_local
LOG=/root/autodl-tmp/newIdea_extracted/newIdea/run/serial_train.log

cd "$ROOT"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

wait_gpu_idle() {
    # 等待 GPU 无计算进程（最多 10 分钟）
    for _ in $(seq 1 120); do
        n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
        if [ "$n" -eq 0 ]; then return 0; fi
        sleep 5
    done
    log "WARN: GPU 10min 内未完全空闲，强制继续"
}

run_task() {
    local name="$1" cfg="$2"
    log "=== Start $name: $cfg ==="
    if ! $PY train_agiqa3k.py --cfg-path "$cfg" --seed 42 --num_cv 1 2>&1 | tee -a "$LOG"; then
        log "!!! $name FAILED"
        return 1
    fi
    log "=== $name DONE ==="
    return 0
}

log "===== serial_train started ====="

# AGIQA-3K 消融（4 行，100ep）
run_task "A0 IP-IQA baseline"        "ipiqa/projects/agiqa3k/ipiqa_ablation_A0.yaml"        || exit 1
wait_gpu_idle; sleep 60
run_task "A1 +Multi-Scale"           "ipiqa/projects/agiqa3k/ipiqa_ablation_A1_ms.yaml"     || exit 1
wait_gpu_idle; sleep 60
run_task "A2 +Dual Attention"        "ipiqa/projects/agiqa3k/ipiqa_ablation_A2_ms_da.yaml"  || exit 1
wait_gpu_idle; sleep 60
run_task "A3 Full (AGIQA-3K)"        "ipiqa/projects/agiqa3k/ipiqa_ablation_A3_full.yaml"   || exit 1

# 第二数据集完整模型
wait_gpu_idle; sleep 60
run_task "A3 Full (AIGCIQA2023)"     "ipiqa/projects/agiqa3k/ipiqa_aigciqa2023_msda.yaml"    || exit 1

log "===== serial_train ALL DONE ====="
