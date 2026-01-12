import sqlite3
import pickle
import os
import sys
import zlib
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 配置部分
# ---------------------------------------------------------
DB_FOLDER = "opentuner.db"
OUTPUT_IMAGE = "opentuner_hv_history.png"

# ---------------------------------------------------------
# 1. 自动定位数据库文件
# ---------------------------------------------------------
try:
    if not os.path.exists(DB_FOLDER):
         raise FileNotFoundError
    db_filename = [f for f in os.listdir(DB_FOLDER) if f.endswith('.db')][0]
except (IndexError, FileNotFoundError):
    print(f"错误：在 {DB_FOLDER} 目录中找不到 .db 文件")
    sys.exit(1)

db_path = os.path.join(DB_FOLDER, db_filename)
print(f"正在读取数据库: {db_path}\n")

# ---------------------------------------------------------
# 2. 连接数据库 & 查询
# ---------------------------------------------------------
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 修改点：按照 collection_date (收集时间) 正序排列
sql = """
SELECT 
    r.id, 
    r.time, 
    r.accuracy, 
    c.data,
    r.collection_date
FROM 
    result r
JOIN 
    configuration c ON r.configuration_id = c.id
WHERE 
    r.state = 'OK' 
ORDER BY 
    r.collection_date ASC;
"""

cursor.execute(sql)
rows = cursor.fetchall()
conn.close()

# ---------------------------------------------------------
# 3. 数据处理与打印
# ---------------------------------------------------------
print(f"{'Order':<5} | {'ID':<4} | {'RPS':<10} | {'Recall':<8} | {'HV (Obj)':<10} | {'Index Type':<12} | {'Params Summary'}")
print("-" * 120)

history_best_hv = []
current_max_hv = 0.0
valid_iterations = []

for idx, row in enumerate(rows):
    run_id = row[0]
    time_val = row[1]   # OpenTuner 存的是 -RPS
    recall = row[2]
    blob_data = row[3]
    
    # --- A. 反序列化配置 ---
    config_dict = {}
    try:
        config_dict = pickle.loads(blob_data)
    except Exception:
        try:
            decompressed_data = zlib.decompress(blob_data)
            config_dict = pickle.loads(decompressed_data)
        except Exception:
            config_dict = {"error": "decode_fail"}

    # --- B. 计算指标 ---
    # 处理 RPS 为无穷大的异常情况
    if time_val == float('inf') or time_val == float('-inf'):
        real_rps = 0.0
    else:
        real_rps = -time_val # 还原 RPS
    
    # 计算超体积 Hypervolume = Recall * RPS
    hypervolume = real_rps * recall
    
    # --- C. 维护历史最优 (Monotonic Increase) ---
    if hypervolume > current_max_hv:
        current_max_hv = hypervolume
    
    history_best_hv.append(current_max_hv)
    valid_iterations.append(idx + 1)

    # --- D. 打印表格 ---
    # 提取索引类型
    index_type = config_dict.get('index_type', 'Unknown')
    
    # 简化打印：只打印除了 index_type 之外的前3个参数，防止表格太长
    # 实际写论文时可以把 config_dict 全部打出来
    simple_params = {k: v for k, v in config_dict.items() if k != 'index_type'}
    param_str = str(simple_params)
    if len(param_str) > 50:
        param_str = param_str[:47] + "..."

    print(f"{idx+1:<5} | {run_id:<4} | {real_rps:<10.2f} | {recall:<8.4f} | {hypervolume:<10.2f} | {index_type:<12} | {param_str}")

# ---------------------------------------------------------
# 4. 绘图 (仿照 VDTuner 风格)
# ---------------------------------------------------------
if len(history_best_hv) > 0:
    plt.figure(figsize=(10, 6))
    
    plt.plot(valid_iterations, history_best_hv, 
             color='red',           # 红色
             linestyle='--',        # 虚线
             marker='s',            # 方块标记
             markersize=4,
             label='OpenTuner Best HV')
    
    plt.ylim(ymin=0) 
    plt.title('OpenTuner Optimization Progress', fontsize=16, fontweight='bold')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Hypervolume (RPS * Recall)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')
    
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print("\n" + "="*50)
    print(f"绘图完成！收敛曲线已保存为: {OUTPUT_IMAGE}")
    print(f"最终最优超体积: {current_max_hv:.2f}")
    print("="*50)
else:
    print("\n没有读取到有效数据，无法绘图。")