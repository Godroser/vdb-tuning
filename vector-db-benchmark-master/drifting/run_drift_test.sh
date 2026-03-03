#!/usr/bin/env bash
#
# 数据漂移性能测试脚本
# 验证随着数据插入和删除（数据漂移），向量数据库原有 knob 配置是否出现性能下降。
#
# 流程：
# 1. 导入指定数据集
# 2. 运行初始性能测试
# 3. 循环：插入新向量、删除旧向量，每隔指定操作数后再次测试性能
#
# 注意：不修改原始数据集文件，新向量在内存中随机生成
#
# 用法（在 drifting 目录或项目根目录执行）：
#   ./drifting/run_drift_test.sh [SERVER_PATH] [ENGINE_NAME] [DATASET] [BATCH_SIZE] [NUM_CYCLES]
#
# 示例：
#   ./drifting/run_drift_test.sh milvus-single-node milvus-p10 random-geo-radius-2048-angular-no-filters 1000 10
#

set -e

# drifting 目录（脚本所在目录）
DRIFTING_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# 项目根目录 = vector-db-benchmark-master
BENCHMARK_ROOT=$(cd "$DRIFTING_DIR/.." && pwd)

SERVER_PATH=${1:-"milvus-single-node"}
ENGINE_NAME=${2:-"milvus-p10"}
DATASETS=${3:-"random-geo-radius-2048-angular-no-filters"}
BATCH_SIZE=${4:-1000}
NUM_CYCLES=${5:-10}
SERVER_HOST="127.0.0.1"

MILVUS_DIR="$BENCHMARK_ROOT/engine/servers/$SERVER_PATH"
MONITOR_DIR="$BENCHMARK_ROOT/monitoring"
DRIFT_STATE_FILE="$DRIFTING_DIR/.drift_state.json"
DRIFT_RESULTS_FILE="$BENCHMARK_ROOT/results/drift_test_results.jsonl"
DRIFT_SEGMENT_STATS_DIR="$BENCHMARK_ROOT/results/drift_segment_stats"

# 虚拟环境
VENV_PATH=${VENV_PATH:-"/talas-pool/home/z78ding/venv"}
if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
fi

if [ -f "$VENV_PATH/bin/python3.12" ]; then
    PYTHON_CMD="$VENV_PATH/bin/python3.12"
elif [ -f "$VENV_PATH/bin/python3" ]; then
    PYTHON_CMD="$VENV_PATH/bin/python3"
else
    PYTHON_CMD="python3.11"
fi

echo "======================================="
echo "📊 数据漂移性能测试"
echo "Engine: $ENGINE_NAME | Dataset: $DATASETS"
echo "Batch: $BATCH_SIZE 向量/轮 | 漂移轮数: $NUM_CYCLES"
echo "======================================="

# 1. 启动监控
echo ">>> [Step 1] 启动后台监控..."
if [ -f "$MONITOR_DIR/monitor_docker.sh" ]; then
    rm -f "$MONITOR_DIR/docker.stats.jsonl"
    nohup bash -c "cd $MONITOR_DIR && ./monitor_docker.sh" > /dev/null 2>&1 &
    MONITOR_PID=$!
    echo "    监控进程 PID: $MONITOR_PID"
else
    echo "⚠️  未找到监控脚本，跳过。"
fi

# 2. 重置 Milvus
echo ">>> [Step 2] 重置 Milvus..."
cd "$MILVUS_DIR"
docker-compose down -v
sleep 5

CLEAN_HOST_VOLUMES=${CLEAN_HOST_VOLUMES:-1}
if [ "$CLEAN_HOST_VOLUMES" = "1" ]; then
    DEFAULT_VOLUME_ROOT="$BENCHMARK_ROOT/engine/servers/$SERVER_PATH/volumes"
    VOLUME_ROOT="${VOLUME_ROOT:-$DEFAULT_VOLUME_ROOT}"
    VOLUME_ROOT_ABS="$(realpath -m "$VOLUME_ROOT" 2>/dev/null || echo "")"
    if [ -n "$VOLUME_ROOT_ABS" ] && { [ "$VOLUME_ROOT_ABS" = "$DEFAULT_VOLUME_ROOT" ] || [[ "$VOLUME_ROOT_ABS" == */milvus-single-node/volumes ]]; }; then
        echo "    🧹 清理宿主机数据目录: $VOLUME_ROOT_ABS"
        sudo rm -rf "${VOLUME_ROOT_ABS:?}/"*
    fi
fi

docker-compose up -d

# 3. 等待服务就绪
echo ">>> [Step 3] 等待 Milvus 启动 (最多 120s)..."
MAX_WAIT=120
WAIT_COUNT=0
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if ! docker ps | grep -q milvus-standalone; then
        echo "    ❌ milvus-standalone 未运行"
        exit 1
    fi
    HEALTH=$(docker inspect --format='{{.State.Health.Status}}' milvus-standalone 2>/dev/null || echo "none")
    if [ "$HEALTH" = "healthy" ]; then
        echo "    ✅ Milvus 已就绪"
        break
    fi
    sleep 2
    WAIT_COUNT=$((WAIT_COUNT + 2))
    [ $((WAIT_COUNT % 10)) -eq 0 ] && echo "    等待中... (${WAIT_COUNT}s/${MAX_WAIT}s)"
done

if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
    echo "    ⚠️  等待超时，继续尝试..."
fi

# 4. 初始导入 + 性能测试
echo ">>> [Step 4] 导入数据并运行初始性能测试..."
cd "$BENCHMARK_ROOT"
export no_proxy="localhost,127.0.0.1,::1"

$PYTHON_CMD run.py --engines "$ENGINE_NAME" --datasets "$DATASETS" --host "$SERVER_HOST"

# 获取初始数据集大小，初始化漂移状态
INITIAL_COUNT=$($PYTHON_CMD "$DRIFTING_DIR/run_drift_cycle.py" --get-initial-count "$DATASETS")
echo "    初始向量数: $INITIAL_COUNT"

echo '{"cycle":0,"base_id":0,"max_id":'$((INITIAL_COUNT - 1))'}' > "$DRIFT_STATE_FILE"

# 提取并记录初始结果
mkdir -p "$BENCHMARK_ROOT/results" "$DRIFT_SEGMENT_STATS_DIR"
rm -f "$DRIFT_RESULTS_FILE"
RES_FILE=$(ls -t "$BENCHMARK_ROOT/results/" 2>/dev/null | grep -E "${ENGINE_NAME}.*${DATASETS}.*search" | head -n 1)
if [ -n "$RES_FILE" ]; then
    echo "{\"cycle\": 0, \"file\": \"$RES_FILE\"}" >> "$DRIFT_RESULTS_FILE"
fi

# 初始测试后统计各 segment 向量数量
echo "    [Cycle 0] 统计各 segment 向量数量..."
$PYTHON_CMD "$DRIFTING_DIR/run_drift_cycle.py" --stats-only \
    --output-stats "$DRIFT_SEGMENT_STATS_DIR/cycle_0.json" 2>/dev/null || true

# 5. 漂移循环
echo ">>> [Step 5] 开始数据漂移循环 ($NUM_CYCLES 轮)..."

for i in $(seq 1 "$NUM_CYCLES"); do
    echo ""
    echo "--- 漂移轮次 $i/$NUM_CYCLES ---"
    
    # 5a. 执行插入+删除
    echo "    [${i}] 插入 $BATCH_SIZE 新向量，删除 $BATCH_SIZE 旧向量..."
    $PYTHON_CMD "$DRIFTING_DIR/run_drift_cycle.py" \
        --dataset "$DATASETS" \
        --batch-size "$BATCH_SIZE" \
        --state-file "$DRIFT_STATE_FILE"
    
    # 等待 Milvus 消化变更
    sleep 5
    
    # 5b. 仅运行搜索测试（不重新上传）
    echo "    [${i}] 运行性能测试..."
    $PYTHON_CMD "$BENCHMARK_ROOT/run.py" --engines "$ENGINE_NAME" --datasets "$DATASETS" --host "$SERVER_HOST" --skip-upload
    
    # 记录结果
    RES_FILE=$(ls -t "$BENCHMARK_ROOT/results/" 2>/dev/null | grep -E "${ENGINE_NAME}.*${DATASETS}.*search" | head -n 1)
    if [ -n "$RES_FILE" ]; then
        echo "{\"cycle\": $i, \"file\": \"$RES_FILE\"}" >> "$DRIFT_RESULTS_FILE"
    fi

    # 统计各 segment 向量数量
    echo "    [${i}] 统计各 segment 向量数量..."
    $PYTHON_CMD "$DRIFTING_DIR/run_drift_cycle.py" --stats-only \
        --output-stats "$DRIFT_SEGMENT_STATS_DIR/cycle_${i}.json" 2>/dev/null || true
done

# 6. 收尾
echo ""
echo ">>> [Step 6] 收尾..."

if [ -n "$MONITOR_PID" ]; then
    kill $MONITOR_PID 2>/dev/null || true
    mkdir -p "$MONITOR_DIR/results"
    LOG_NAME=$(echo "$ENGINE_NAME" | sed 's/[^A-Za-z0-9._-]/_/g')
    mv "$MONITOR_DIR/docker.stats.jsonl" "$MONITOR_DIR/results/${LOG_NAME}-drift-docker.stats.jsonl" 2>/dev/null || true
fi

echo ""
echo "======================================="
echo "📊 数据漂移测试完成"
echo "结果文件: $DRIFT_RESULTS_FILE"
echo "Segment 统计: $DRIFT_SEGMENT_STATS_DIR/cycle_*.json"
echo "可对比各 cycle 的 search 结果及 segment 向量分布分析性能变化"
echo "======================================="
