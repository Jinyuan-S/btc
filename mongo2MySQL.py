import pymongo
import pymysql

# ========== 配置部分 ==========
MONGO_URI = "mongodb://localhost:27017"
MONGO_DB  = "btc"           # MongoDB数据库
ADDR_COLL = "addr"          # btc.addr 集合
SNAP_COLL = "snap"          # btc.snap 集合

MYSQL_HOST     = "127.0.0.1"
MYSQL_PORT     = 3306
MYSQL_USER     = "root"
MYSQL_PASSWORD = "gsj123"
MYSQL_DB       = "btc"   # 你在MySQL中创建的数据库

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
# 1. 将地址数据 (btc.addr) 导入 addresses 表
# =====================================
print(">>> Importing addresses ...")

addr_docs = coll_addr.find({})
count_addr = 0

for doc in addr_docs:
    # 从Mongo文档提取字段
    # 如果Mongo里没有某个字段，就给个默认值
    keyhash     = str(doc['_id'])         # 或者 doc['_id'].hex() 视情况
    addr_str    = doc.get('addr', None)
    type_int    = doc.get('type', -1)
    val_satoshi = doc.get('val', 0)
    key_seen    = doc.get('key-seen', 0)
    ins_count   = doc.get('ins', 0)       # 如果在 dat2mongo.py 中有存储
    outs_count  = doc.get('outs', 0)
    last_height = doc.get('height', None)

    # 构造SQL插入
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
    # 上面用 ON DUPLICATE KEY UPDATE 以便“upsert”效果，
    # 前提是你给 addresses(keyhash) 建了唯一索引(UNIQUE keyhash)。

    cursor.execute(sql_addr, (
        keyhash, addr_str, type_int, val_satoshi,
        key_seen, ins_count, outs_count, last_height
    ))
    count_addr += 1

conn.commit()
print(f"Done addresses: {count_addr} rows inserted/updated.")


# =====================================
# 2. 导入快照数据 (btc.snap) -> snapshots, snapshot_summary_by_type, snapshot_quantum_by_type
# =====================================
print(">>> Importing snapshots ...")

snap_docs = coll_snap.find({})
count_snap = 0
count_summary = 0
count_quantum = 0

for doc in snap_docs:
    height    = doc['_id']        # int
    snap_date = doc.get('date', None)
    # 这里snap_date可能是一个 datetime对象 或者字符串，看dat2mongo.py写入情况:
    # 如果需要datetime，可能要做转换: snap_date.isoformat() 或者 str(snap_date)

    tot_val   = doc.get('tot-val', 0)
    op_return = doc.get('op-return', 0)
    unknown   = doc.get('unknown', 0)
    qattack   = doc.get('qattack-frac', 0.0)
    un_frac   = doc.get('unknown-frac', 0.0)

    # 插入 snapshots
    sql_snap = """
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
    cursor.execute(sql_snap, (
        height, snap_date, tot_val, op_return,
        unknown, qattack, un_frac
    ))
    count_snap += 1

    # ---------- summary-by-type ----------
    summary_array = doc.get('summary-by-type', [])
    for item in summary_array:
        addr_type = item.get('type', 'unknown')
        num_pos   = item.get('num-pos', 0)
        tv        = item.get('tot-val', 0)

        sql_sum = """
        INSERT INTO snapshot_summary_by_type
          (snap_height, addr_type, num_pos, tot_val)
        VALUES
          (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
          num_pos=VALUES(num_pos),
          tot_val=VALUES(tot_val)
        """
        cursor.execute(sql_sum, (height, addr_type, num_pos, tv))
        count_summary += 1

    # ---------- quantum-by-type ----------
    quantum_array = doc.get('quantum-by-type', [])
    for item in quantum_array:
        addr_type = item.get('type', 'unknown')
        num_pos   = item.get('num-pos', 0)
        tv        = item.get('tot-val', 0)

        sql_q = """
        INSERT INTO snapshot_quantum_by_type
          (snap_height, addr_type, num_pos, tot_val)
        VALUES
          (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
          num_pos=VALUES(num_pos),
          tot_val=VALUES(tot_val)
        """
        cursor.execute(sql_q, (height, addr_type, num_pos, tv))
        count_quantum += 1

conn.commit()
print(f"Done snapshots: {count_snap} rows.")
print(f"Done summary-by-type: {count_summary} rows.")
print(f"Done quantum-by-type: {count_quantum} rows.")

cursor.close()
conn.close()
mongo_client.close()

print(">>> All done.")
