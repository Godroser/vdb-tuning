import sqlite3
import pickle
import os
import sys
import zlib

# ---------------------------------------------------------
# 1. 自动定位数据库文件
# ---------------------------------------------------------
db_folder = "opentuner.db"
try:
    if not os.path.exists(db_folder):
         raise FileNotFoundError
    # 找到文件夹里唯一的 .db 文件
    db_filename = [f for f in os.listdir(db_folder) if f.endswith('.db')][0]
except (IndexError, FileNotFoundError):
    print(f"错误：在 {db_folder} 目录中找不到 .db 文件")
    sys.exit(1)

db_path = os.path.join(db_folder, db_filename)
print(f"正在读取数据库: {db_path}\n")

# ---------------------------------------------------------
# 2. 连接数据库 & 查询
# ---------------------------------------------------------
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 联合查询 result 和 configuration
# 按照 ID 排序，这样能看到 OpenTuner 是如何一步步尝试的
sql = """
SELECT 
    r.id, 
    r.time, 
    c.data 
FROM 
    result r
JOIN 
    configuration c ON r.configuration_id = c.id
WHERE 
    r.state = 'OK'
ORDER BY 
    r.id ASC;
"""

cursor.execute(sql)
rows = cursor.fetchall()
conn.close()

# ---------------------------------------------------------
# 3. 数据处理与打印
# ---------------------------------------------------------
print(f"{'Run ID':<8} | {'Param X':<10} | {'Result Y (Time)':<20}")
print("-" * 50)

best_y = float('inf')
best_x = None

for row in rows:
    run_id = row[0]
    y_val = row[1]   # 在 SquareTuner 里，time 就是 y = (x-5)^2
    blob_data = row[2]

    # --- 反序列化配置 (解码 X) ---
    config_dict = {}
    try:
        config_dict = pickle.loads(blob_data)
    except Exception:
        try:
            # 尝试 zlib 解压
            decompressed_data = zlib.decompress(blob_data)
            config_dict = pickle.loads(decompressed_data)
        except Exception as e:
            print(f"ID {run_id} 解析失败: {e}")
            continue

    # 提取 X
    x_val = config_dict.get('x', 'N/A')

    # 打印一行数据
    print(f"{run_id:<8} | {x_val:<10} | {y_val:<20.4f}")

    # 记录最优解
    if isinstance(y_val, (int, float)) and y_val < best_y:
        best_y = y_val
        best_x = x_val

print("-" * 50)
print(f"【总结】\nOpenTuner 找到的最小值 (Best Y): {best_y}")
print(f"对应的参数 (Best X): {best_x}")