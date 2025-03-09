import pymongo
import pymysql
import math
import os

# ========== 配置部分 ==========
MONGO_URI = "mongodb://localhost:27017"
MONGO_DB  = "btc"           # MongoDB数据库
ADDR_COLL = "addr"          # btc.addr 集合
SNAP_COLL = "snap"          # btc.snap 集合

MYSQL_HOST     = "127.0.0.1"
MYSQL_PORT     = 3306
MYSQL_USER     = "root"
MYSQL_PASSWORD = "btcbtc"
MYSQL_DB       = "btc_3"   # 你在MySQL中创建的数据库

# 批量提交的大小
BATCH_SIZE = 2000

# 进度文件名
PROGRESS_ADDR_FILE = "progress_addr.txt"
PROGRESS_SNAP_FILE = "progress_snap.txt"

# ========== 辅助函数：读取 / 保存进度 ==========

def load_progress(filename):
    """从本地文件读取已处理文档数量，若无则返回0"""
    if not os.path.exists(filename):
        return 0
    with open(filename, 'r', encoding='utf-8') as f:
        val = f.read().strip()
        if val.isdigit():
            return int(val)
        return 0

def save_progress(filename, count):
    """把已处理文档数写入本地文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(str(count))

# ========== 连接MongoDB ==========
mongo_client = pymongo.MongoClient(MONGO_URI)
mdb = mongo_client[MONGO_DB]

coll_addr = mdb[ADDR_COLL]
coll_snap = mdb[SNAP_COLL]

# ========== 连接MySQL ==========
conn = pymysql.connect(
    host=MYSQL_HOST,
    port=MYSQL_PORT,
    user=MYSQL_USER,
    password=MYSQL_PASSWORD,
    database=MYSQL_DB,
    charset='utf8mb4'
)
cursor = conn.cursor()

# =====================================
# 1. 导入 addresses
# =====================================
print(">>> Importing addresses ...")

# 获取文档总数（若数量极大，也可以用 estimated_document_count()）
addr_count_total = coll_addr.count_documents({})
print(f"Total addresses to process: {addr_count_total}")

# 读取上次的进度(已经处理多少条)
progress_addr = load_progress(PROGRESS_ADDR_FILE)
print(f"Resuming from addresses skip={progress_addr} ...")

# 用skip跳过已经处理的文档
addr_cursor = coll_addr.find({}).skip(progress_addr)
count_addr = progress_addr  # 表示我们已经跳过了这部分

batch_data = []

sql_addr = """
INSERT INTO addresses
  (keyhash, addr, type, val, key_seen, ins_count, outs_count, last_height)
VALUES
  (%s, %s, %s, %s, %s, %s, %s, %s)
ON DUPLICATE KEY UPDATE
  val=VALUES(val),
  key_seen=VALUES(key_seen),
  ins_count=VALUES(ins_count),
  outs_count=VALUES(outs_count),
  last_height=VALUES(last_height)
"""

for doc in addr_cursor:
    # 提取字段
    keyhash     = str(doc['_id'])
    addr_str    = doc.get('addr', None)
    type_int    = doc.get('type', -1)
    val_satoshi = doc.get('val', 0)
    key_seen    = doc.get('key-seen', 0)
    ins_count   = doc.get('ins', 0)
    outs_count  = doc.get('outs', 0)
    last_height = doc.get('height', None)

    row = (
        keyhash, addr_str, type_int, val_satoshi,
        key_seen, ins_count, outs_count, last_height
    )
    batch_data.append(row)
    count_addr += 1

    # 如果达到了批大小，执行一次批量插入
    if len(batch_data) >= BATCH_SIZE:
        cursor.executemany(sql_addr, batch_data)
        conn.commit()
        batch_data.clear()

        # 保存进度
        save_progress(PROGRESS_ADDR_FILE, count_addr)

        # 打印进度
        progress_percent = count_addr / addr_count_total * 100 if addr_count_total else 0
        print(f"Processed {count_addr} / {addr_count_total} ({progress_percent:.2f}%) addresses", flush=True)

# 处理剩余不满一批的数据
if batch_data:
    cursor.executemany(sql_addr, batch_data)
    conn.commit()
    batch_data.clear()
    # 保存进度
    save_progress(PROGRESS_ADDR_FILE, count_addr)

print(f"Done addresses: {count_addr} rows inserted/updated.\n")


# =====================================
# 2. 导入快照 snapshots
# =====================================
print(">>> Importing snapshots ...")

snap_count_total = coll_snap.count_documents({})
print(f"Total snapshots to process: {snap_count_total}")

snap_sql = """
INSERT INTO snapshots
  (height, snap_date, tot_val, op_return, unknown, qattack_frac, unknown_frac)
VALUES
  (%s, %s, %s, %s, %s, %s, %s)
ON DUPLICATE KEY UPDATE
  snap_date=VALUES(snap_date),
  tot_val=VALUES(tot_val),
  op_return=VALUES(op_return),
  unknown=VALUES(unknown),
  qattack_frac=VALUES(qattack_frac),
  unknown_frac=VALUES(unknown_frac)
"""

sql_sum = """
INSERT INTO snapshot_summary_by_type
  (snap_height, addr_type, num_pos, tot_val)
VALUES
  (%s, %s, %s, %s)
ON DUPLICATE KEY UPDATE
  num_pos=VALUES(num_pos),
  tot_val=VALUES(tot_val)
"""

sql_q = """
INSERT INTO snapshot_quantum_by_type
  (snap_height, addr_type, num_pos, tot_val)
VALUES
  (%s, %s, %s, %s)
ON DUPLICATE KEY UPDATE
  num_pos=VALUES(num_pos),
  tot_val=VALUES(tot_val)
"""

# 读取 snapshots 的进度
progress_snap = load_progress(PROGRESS_SNAP_FILE)
print(f"Resuming from snapshots skip={progress_snap} ...")

snap_cursor = coll_snap.find({}).skip(progress_snap)
count_snap = progress_snap

batch_size_snap = 1000  # 你也可以用同一个BATCH_SIZE
commit_counter = 0

for doc in snap_cursor:
    height = doc['_id']
    snap_date = doc.get('date', None)
    tot_val   = doc.get('tot-val', 0)
    op_return = doc.get('op-return', 0)
    unknown   = doc.get('unknown', 0)
    qattack   = doc.get('qattack-frac', 0.0)
    un_frac   = doc.get('unknown-frac', 0.0)

    # 插入 snapshots
    cursor.execute(snap_sql, (height, snap_date, tot_val, op_return, unknown, qattack, un_frac))

    # summary-by-type
    summary_array = doc.get('summary-by-type', [])
    for item in summary_array:
        addr_type = item.get('type', 'unknown')
        num_pos   = item.get('num-pos', 0)
        tv        = item.get('tot-val', 0)
        cursor.execute(sql_sum, (height, addr_type, num_pos, tv))

    # quantum-by-type
    quantum_array = doc.get('quantum-by-type', [])
    for item in quantum_array:
        addr_type = item.get('type', 'unknown')
        num_pos   = item.get('num-pos', 0)
        tv        = item.get('tot-val', 0)
        cursor.execute(sql_q, (height, addr_type, num_pos, tv))

    count_snap += 1
    commit_counter += 1

    # 每过一定数量commit一次，并保存进度
    if commit_counter >= batch_size_snap:
        conn.commit()
        save_progress(PROGRESS_SNAP_FILE, count_snap)

        progress_percent = count_snap / snap_count_total * 100 if snap_count_total else 0
        print(f"Processed {count_snap} / {snap_count_total} snapshots ({progress_percent:.2f}%)", flush=True)

        commit_counter = 0

# 提交剩余
conn.commit()
save_progress(PROGRESS_SNAP_FILE, count_snap)

print(f"Done snapshots: {count_snap} rows.\n")

cursor.close()
conn.close()
mongo_client.close()

print(">>> All done.")
