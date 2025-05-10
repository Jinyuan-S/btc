

"""
dat2MySQL.py

【分阶段方案】：
  1. 并行解析 dat 文件，每批处理一定数量（例如 4 个）dat 文件，
     将解析结果逐条写入临时 CSV 文件（逐条写入以降低内存占用），
     并在每个进程中每 5% 打印一次进度提示。
  2. 按 dat 文件顺序加载 CSV 数据到 MySQL 中（使用 LOAD DATA LOCAL INFILE）。
  
要求：
  - 完整解析 dat 文件内所有信息（区块、交易、输入、输出、地址）。
  - 支持断点续传：checkpoint 记录下一个待处理文件及 offset。
  - 并行解析阶段采用多进程，每批处理固定数量（例如 4 个）dat 文件，
    每个进程定期打印简要进度（每 5% 一次）。
  - 加载阶段严格按 dat 文件顺序加载 CSV 数据到数据库中，保证数据顺序正确。
  - 每个 dat 文件处理完毕后，checkpoint 自动更新为下一个文件（格式保持一致）且 offset 置 0；
    如果下一个文件不存在，则提示用户手动添加后退出。
  - 为加速 CSV 到 MySQL 的插入，在加载阶段禁用外键和唯一性检查，加载完成后恢复。
  - 当某个 dat 文件完全加载完毕后，程序将倒计时 5 秒提示用户是否继续处理下一个文件：
       用户输入 “Y” (不区分大小写) 并回车则继续处理，
       输入其它或提前输入则程序终止，并清理所有临时 CSV 文件。
       若 5 秒内无输入，则自动继续。

[Phased scheme] :
    1. Parse dat files in parallel, processing a certain number (e.g., 4) dat files in a batch, 
    writing the parsing results to a temporary CSV file one by one (to reduce memory usage), 
    and printing progress notes every 5% of each process.
    2. Load the CSV DATA into MySQL in dat file order (using LOAD DATA LOCAL INFILE).

Requirements:
    - Fully parse all the information in dat (blocks, transactions, inputs, outputs, addresses).
    - Support for resume at breakpoints: checkpoint records the next pending file and the offset.
    - The parallel parsing phase employs multiple processes that process a fixed number of (say, four) dat files per batch, and each process periodically prints a summary progress (every 5%).
    - The loading phase loads CSV data into the database strictly according to the dat file order to ensure the correct data order.
    - As each dat file is processed, checkpoint automatically updates to the next file (in the same format) with offset set to 0. 
        If the next file does not exist, the user is prompted to add it manually and exit.
    - To speed up CSV insertion into MySQL, foreign keys and uniqueness checks are disabled during the loading phase and restored after loading is complete.
    - When a dat file is fully loaded, the program will count down to 5 seconds to prompt the user whether to proceed to the next file:
        The user enters "Y" (case insensitive) and presses enter to continue processing,
        Enter something else or enter early and the program terminates and clears all temporary CSV files.
        If there is no input within 5 seconds, it will continue automatically.
"""

import os
import sys
import json
import struct
import logging
import hashlib
import mysql.connector
from mysql.connector import errorcode
from io import BytesIO
import csv
from multiprocessing import Pool, current_process
import threading
import time
sys.stdout.reconfigure(encoding='gbk')
# ====== 配置区 / Configuration area ======
DATA_DIR = "E:/Prof Marco FYP Blockchain/data/blocks"          # 存放 blk*.dat 文件的目录 / The directory to store the blk*.dat file
CHECKPOINT_FILE = "checkpoint.json"
DB_CONFIG = {
    "user": "root",
    "password": "btcbtc",
    "host": "127.0.0.1",
    "port": 3306,
    "database": "btc_analysis_2",
}
NUM_PROCESSES = 4          # 并行进程数 / Number of parallel processes
BATCH_SIZE = 4             # 每批处理 dat 文件数量 / Number of dat files processed per batch

# 用于生成 CSV 的批量参数（较大值可减少 LOAD DATA 调用次数） / Batch parameter for CSV generation (higher values reduce LOAD DATA calls)
BATCH_SIZE_BLOCKS = 10000
BATCH_SIZE_TX = 10000
BATCH_SIZE_INPUT = 10000
BATCH_SIZE_OUTPUT = 10000
BATCH_SIZE_ADDRESS = 5000

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# ====== CSV 写入函数 / CSV write function ======
def write_csv(filename, rows):
    """将 rows 写入 CSV 文件 / Write rows to a CSV file"""
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        for row in rows:
            writer.writerow(row)

# ====== 清理临时 CSV 文件 / Clean up temporary CSV files ======
def cleanup_temp_csv():
    """删除当前目录下所有以 'temp_' 开头、'.csv' 结尾的文件 / Delete all files in the current directory that start with 'temp_' and end with '.csv'"""
    for file in os.listdir('.'):
        if file.startswith('temp_') and file.endswith('.csv'):
            try:
                os.remove(file)
                logging.info(f"Delete temporary files: {file}")
            except Exception as e:
                logging.error(f"Failure to delete {file} temporary file: {e}")

# ====== Base58 与 Bech32 辅助函数 / Base58 and Bech32 helper functions ======
ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"

def base58_encode(b: bytes) -> str:
    num = int.from_bytes(b, byteorder='big')
    result = ""
    while num > 0:
        num, mod = divmod(num, 58)
        result = ALPHABET[mod] + result
    n_pad = 0
    for byte in b:
        if byte == 0:
            n_pad += 1
        else:
            break
    return "1" * n_pad + result

def base58_check_encode(b: bytes) -> str:
    checksum = double_sha256(b)[:4]
    return base58_encode(b + checksum)

CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"

def bech32_polymod(values):
    GENERATOR = [0x3b6a57b2, 0x26508e6d, 0x1ea119fa, 0x3d4233dd, 0x2a1462b3]
    chk = 1
    for v in values:
        b = chk >> 25
        chk = ((chk & 0x1ffffff) << 5) ^ v
        for i in range(5):
            if (b >> i) & 1:
                chk ^= GENERATOR[i]
    return chk

def bech32_hrp_expand(hrp):
    return [ord(x) >> 5 for x in hrp] + [0] + [ord(x) & 31 for x in hrp]

def bech32_create_checksum(hrp, data):
    values = bech32_hrp_expand(hrp) + data
    polymod = bech32_polymod(values + [0,0,0,0,0,0]) ^ 1
    return [(polymod >> 5 * (5 - i)) & 31 for i in range(6)]

def convertbits(data, frombits, tobits, pad=True):
    acc = 0
    bits = 0
    ret = []
    maxv = (1 << tobits) - 1
    for value in data:
        if value < 0 or (value >> frombits):
            return None
        acc = (acc << frombits) | value
        bits += frombits
        while bits >= tobits:
            bits -= tobits
            ret.append((acc >> bits) & maxv)
    if pad:
        if bits:
            ret.append((acc << (tobits - bits)) & maxv)
    elif bits >= frombits or ((acc << (tobits - bits)) & maxv):
        return None
    return ret

def bech32_encode(hrp, witver, witprog):
    data = [witver] + convertbits(witprog, 8, 5, True)
    checksum = bech32_create_checksum(hrp, data)
    combined = data + checksum
    return hrp + "1" + "".join([CHARSET[d] for d in combined])

# ====== 定义 read_varint / Define read_varint ======
def read_varint(stream: BytesIO) -> int:
    i = stream.read(1)
    if len(i) == 0:
        raise EOFError("EOF while reading varint")
    i = i[0]
    if i < 0xfd:
        return i
    elif i == 0xfd:
        data = stream.read(2)
        return int.from_bytes(data, 'little')
    elif i == 0xfe:
        data = stream.read(4)
        return int.from_bytes(data, 'little')
    else:
        data = stream.read(8)
        return int.from_bytes(data, 'little')

# ====== 计算双 SHA256 / Calculate double SHA256 ======
def double_sha256(data: bytes) -> bytes:
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()

# ====== 地址提取函数 / Address extraction function ======
def extract_address(script_hex: str):
    if script_hex.startswith("76a914") and script_hex.endswith("88ac") and len(script_hex) == 50:
        hash160 = bytes.fromhex(script_hex[6:46])
        addr = base58_check_encode(b'\x00' + hash160)
        return addr, "P2PKH"
    elif script_hex.startswith("a914") and script_hex.endswith("87") and len(script_hex) == 46:
        hash160 = bytes.fromhex(script_hex[4:44])
        addr = base58_check_encode(b'\x05' + hash160)
        return addr, "P2SH"
    elif script_hex.startswith("0014") and len(script_hex) == 44:
        hash160 = bytes.fromhex(script_hex[4:44])
        addr = bech32_encode("bc", 0, list(hash160))
        return addr, "P2WPKH"
    elif script_hex.startswith("0020") and len(script_hex) == 68:
        hash256 = bytes.fromhex(script_hex[4:68])
        addr = bech32_encode("bc", 0, list(hash256))
        return addr, "P2WSH"
    else:
        return None, "unknown"

# ====== 解析交易、区块 / Parse transactions, blocks ======
def parse_transaction(stream: BytesIO) -> (dict, bytes):
    tx_start = stream.tell()
    try:
        version_bytes = stream.read(4)
        if len(version_bytes) < 4:
            raise EOFError("EOF in transaction version")
        version = int.from_bytes(version_bytes, 'little')
        in_count = read_varint(stream)
        inputs = []
        for idx in range(in_count):
            prev_txid_bytes = stream.read(32)
            prev_txid = prev_txid_bytes[::-1].hex()
            prev_index = int.from_bytes(stream.read(4), 'little')
            script_len = read_varint(stream)
            script_sig = stream.read(script_len)
            sequence = int.from_bytes(stream.read(4), 'little')
            inputs.append({
                "prev_txid": prev_txid,
                "prev_index": prev_index,
                "script_sig": script_sig.hex(),
                "sequence": sequence,
            })
        out_count = read_varint(stream)
        outputs = []
        for idx in range(out_count):
            value = int.from_bytes(stream.read(8), 'little')
            script_len = read_varint(stream)
            script_pub_key = stream.read(script_len)
            outputs.append({
                "value": value,
                "script_pub_key": script_pub_key.hex(),
            })
        lock_time_bytes = stream.read(4)
        if len(lock_time_bytes) < 4:
            raise EOFError("EOF in transaction lock_time")
        lock_time = int.from_bytes(lock_time_bytes, 'little')
    except Exception as e:
        logging.error(f"Error parsing transaction: {e}")
        raise e
    tx_end = stream.tell()
    stream.seek(tx_start)
    tx_raw = stream.read(tx_end - tx_start)
    txid = double_sha256(tx_raw)[::-1].hex()
    tx = {
        "version": version,
        "lock_time": lock_time,
        "inputs": inputs,
        "outputs": outputs,
        "txid": txid,
        "input_count": in_count,
        "output_count": len(outputs),
    }
    return tx, tx_raw

def parse_block(file_obj, file_name: str) -> dict:
    start_offset = file_obj.tell()
    header = file_obj.read(8)
    if len(header) < 8:
        return None
    magic = header[:4]
    if magic != b'\xf9\xbe\xb4\xd9':
        logging.error(f"Invalid magic {magic.hex()} in {file_name} offset {start_offset}")
        return None
    block_length = int.from_bytes(header[4:8], 'little')
    block_data = file_obj.read(block_length)
    if len(block_data) != block_length:
        logging.info(f"Block data is incomplete, in {file_name} offset {start_offset}, May be at the end of the file.")
        return None
    raw_block = header + block_data
    block_size = len(raw_block)
    if len(block_data) < 80:
        logging.info(f"The block data is too short to parse the block header, in {file_name} offset {start_offset}.")
        return None
    block_header = block_data[:80]
    version = int.from_bytes(block_header[0:4], 'little')
    prev_block_hash = block_header[4:36][::-1].hex()
    merkle_root = block_header[36:68][::-1].hex()
    timestamp = int.from_bytes(block_header[68:72], 'little')
    bits = int.from_bytes(block_header[72:76], 'little')
    nonce = int.from_bytes(block_header[76:80], 'little')
    block_hash = double_sha256(block_header)[::-1].hex()
    stream = BytesIO(block_data)
    stream.seek(80)
    try:
        tx_count = read_varint(stream)
    except Exception as e:
        logging.error(f"Failed to read the number of transactions: {e}")
        return None
    transactions = []
    for i in range(tx_count):
        try:
            tx, tx_raw = parse_transaction(stream)
            transactions.append({
                "txid": tx["txid"],
                "version": tx["version"],
                "lock_time": tx["lock_time"],
                "raw_tx": tx_raw,
                "inputs": tx["inputs"],
                "outputs": tx["outputs"],
                "input_count": tx["input_count"],
                "output_count": tx["output_count"],
            })
        except Exception as e:
            logging.error(f"Parse the No.{i} transaction failure: {e}")
            break
    return {
        "block_hash": block_hash,
        "version": version,
        "prev_block_hash": prev_block_hash,
        "merkle_root": merkle_root,
        "timestamp": timestamp,
        "bits": bits,
        "nonce": nonce,
        "block_size": block_size,
        "tx_count": tx_count,
        "raw_block": raw_block,
        "transactions": transactions,
        "file_name": file_name,
        "file_offset": start_offset,
    }

# ====== update_local_address 函数 / update_local_address function ======
def update_local_address(address_dict, addr, received=0, sent=0, block_hash=None, addr_type="unknown", pubkey_flag=0):
    if addr not in address_dict:
        address_dict[addr] = {
            "address": addr,
            "address_type": addr_type,
            "total_received": received,
            "total_sent": sent,
            "balance": received - sent,
            "pubkey_revealed": pubkey_flag,
            "first_seen_block": block_hash,
            "last_seen_block": block_hash,
        }
    else:
        rec = address_dict[addr]
        rec["total_received"] += received
        rec["total_sent"] += sent
        rec["balance"] = rec["total_received"] - rec["total_sent"]
        rec["last_seen_block"] = block_hash
        if pubkey_flag:
            rec["pubkey_revealed"] = 1

# ====== prompt_continue 函数 / The prompt_continue function ======
def prompt_continue(timeout=5):
    """提示用户是否继续处理下一个 dat 文件，等待 timeout 秒
       提示：输入 Y 回车继续，其他输入或超时则终止程序。 / Prompt the user whether to continue processing the next dat file, wait for timeout seconds prompt: 
       Enter Y enter to continue, other input or timeout terminates the program."""
    print("Do you move on to the next file? (Y/N): ", end="", flush=True)
    result = []
    def get_input():
        result.append(sys.stdin.readline().strip())
    t = threading.Thread(target=get_input)
    t.daemon = True
    t.start()
    t.join(timeout)
    if result:
        if result[0].lower() == 'y':
            return True
        else:
            return False
    else:
        return True

# ====== 并行解析阶段：Worker函数 / Parallel parsing phase: Worker function ======
def parse_dat_file(filename):
    blocks_csv = f"temp_{filename}_blocks.csv"
    tx_csv = f"temp_{filename}_tx.csv"
    inputs_csv = f"temp_{filename}_inputs.csv"
    outputs_csv = f"temp_{filename}_outputs.csv"
    address_csv = f"temp_{filename}_address.csv"
    blocks_file = open(blocks_csv, "w", newline="", encoding="utf-8")
    tx_file = open(tx_csv, "w", newline="", encoding="utf-8")
    inputs_file = open(inputs_csv, "w", newline="", encoding="utf-8")
    outputs_file = open(outputs_csv, "w", newline="", encoding="utf-8")
    address_file = open(address_csv, "w", newline="", encoding="utf-8")
    blocks_writer = csv.writer(blocks_file, quoting=csv.QUOTE_MINIMAL)
    tx_writer = csv.writer(tx_file, quoting=csv.QUOTE_MINIMAL)
    inputs_writer = csv.writer(inputs_file, quoting=csv.QUOTE_MINIMAL)
    outputs_writer = csv.writer(outputs_file, quoting=csv.QUOTE_MINIMAL)
    address_writer = csv.writer(address_file, quoting=csv.QUOTE_MINIMAL)

    local_address = {}
    local_utxo = {}
    file_path = os.path.join(DATA_DIR, filename)
    file_size = os.path.getsize(file_path)
    progress_last = 0
    proc_id = current_process().pid
    try:
        with open(file_path, "rb") as f:
            while True:
                cur = f.tell()
                if file_size > 0:
                    progress = (cur / file_size) * 100
                    if progress - progress_last >= 5:
                        logging.info(f"Process {proc_id} Processing files {filename}: {progress:.1f}%")
                        progress_last = progress
                block = parse_block(f, filename)
                if block is None:
                    break
                blocks_writer.writerow([
                    block["block_hash"],
                    block["version"],
                    block["prev_block_hash"],
                    block["merkle_root"],
                    block["timestamp"],
                    block["bits"],
                    block["nonce"],
                    block["block_size"],
                    block["tx_count"],
                    block["raw_block"].hex(),
                    block["file_name"],
                    block["file_offset"],
                ])
                for tx in block["transactions"]:
                    tx_writer.writerow([
                        tx["txid"],
                        block["block_hash"],
                        tx["version"],
                        tx["input_count"],
                        tx["output_count"],
                        tx["lock_time"],
                        tx["raw_tx"].hex(),
                    ])
                    for idx_in, txin in enumerate(tx["inputs"]):
                        inputs_writer.writerow([
                            tx["txid"],
                            idx_in,
                            txin["prev_txid"],
                            txin["prev_index"],
                            txin["script_sig"],
                            txin["sequence"],
                        ])
                        if txin["prev_txid"] != "0"*64:
                            key = (txin["prev_txid"], txin["prev_index"])
                            if key in local_utxo:
                                addr, value = local_utxo.pop(key)
                                update_local_address(local_address, addr, sent=value, block_hash=block["block_hash"])
                    for idx_out, txout in enumerate(tx["outputs"]):
                        outputs_writer.writerow([
                            tx["txid"],
                            idx_out,
                            txout["value"],
                            txout["script_pub_key"],
                        ])
                        addr, addr_type = extract_address(txout["script_pub_key"])
                        if addr:
                            update_local_address(local_address, addr, received=txout["value"], block_hash=block["block_hash"], addr_type=addr_type)
                            local_utxo[(tx["txid"], idx_out)] = (addr, txout["value"])
    except Exception as e:
        logging.error(f"An error occurred parsing the {filename} file: {e}")
        raise e
    for rec in local_address.values():
        address_writer.writerow([
            rec["address"],
            rec["address_type"],
            rec["total_received"],
            rec["total_sent"],
            rec["balance"],
            rec["pubkey_revealed"],
            rec["first_seen_block"],
            rec["last_seen_block"],
        ])
    blocks_file.close()
    tx_file.close()
    inputs_file.close()
    outputs_file.close()
    address_file.close()
    logging.info(f"The {filename} is parsed and a CSV file is generated.")
    return {"blocks": blocks_csv, "tx": tx_csv, "inputs": inputs_csv, "outputs": outputs_csv, "address": address_csv}, filename

# ====== 断点续传相关 / Breakpoint continuation related ======
def load_checkpoint() -> dict:
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r") as f:
                content = f.read().strip()
                if not content:
                    return {"next_file": "blk00000.dat", "offset": 0}
                return json.loads(content)
        except Exception as e:
            logging.error(f"Failed to read checkpoint: {e}")
            return {"next_file": "blk00000.dat", "offset": 0}
    else:
        return {"next_file": "blk00000.dat", "offset": 0}

def save_checkpoint(next_file: str, offset: int):
    checkpoint = {"next_file": next_file, "offset": offset}
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f)

def get_dat_files() -> list:
    files = [f for f in os.listdir(DATA_DIR) if f.startswith("blk") and f.endswith(".dat")]
    files.sort()
    return files

def get_next_filename(current_file: str) -> str:
    try:
        num = int(current_file[3:8])
    except Exception:
        num = 0
    next_num = num + 1
    return f"blk{next_num:05d}.dat"

# ====== SQL 语句模板 / SQL Statement Template ======
INSERT_BLOCK_SQL = (
    "LOAD DATA LOCAL INFILE '{csv_file}' "
    "INTO TABLE blocks "
    "FIELDS TERMINATED BY ',' ENCLOSED BY '\"' ESCAPED BY '\\\\' "
    "LINES TERMINATED BY '\\n' "
    "(block_hash, version, prev_block_hash, merkle_root, timestamp, bits, nonce, block_size, tx_count, @raw_block, file_name, file_offset) "
    "SET raw_block = UNHEX(@raw_block)"
)
INSERT_TX_SQL = (
    "LOAD DATA LOCAL INFILE '{csv_file}' "
    "INTO TABLE transactions "
    "FIELDS TERMINATED BY ',' ENCLOSED BY '\"' ESCAPED BY '\\\\' "
    "LINES TERMINATED BY '\\n' "
    "(txid, block_hash, version, input_count, output_count, lock_time, @raw_tx) "
    "SET raw_tx = UNHEX(@raw_tx)"
)
INSERT_INPUT_SQL = (
    "LOAD DATA LOCAL INFILE '{csv_file}' "
    "INTO TABLE tx_inputs "
    "FIELDS TERMINATED BY ',' ENCLOSED BY '\"' ESCAPED BY '\\\\' "
    "LINES TERMINATED BY '\\n' "
    "(txid, input_index, prev_txid, prev_output_index, @script_sig, sequence) "
    "SET script_sig = UNHEX(@script_sig)"
)
INSERT_OUTPUT_SQL = (
    "LOAD DATA LOCAL INFILE '{csv_file}' "
    "INTO TABLE tx_outputs "
    "FIELDS TERMINATED BY ',' ENCLOSED BY '\"' ESCAPED BY '\\\\' "
    "LINES TERMINATED BY '\\n' "
    "(txid, output_index, value, @script_pub_key) "
    "SET script_pub_key = UNHEX(@script_pub_key)"
)
INSERT_ADDRESS_SQL = (
    "LOAD DATA LOCAL INFILE '{csv_file}' "
    "REPLACE INTO TABLE address "
    "FIELDS TERMINATED BY ',' ENCLOSED BY '\"' ESCAPED BY '\\\\' "
    "LINES TERMINATED BY '\\n' "
    "(address, address_type, total_received, total_sent, balance, pubkey_revealed, first_seen_block, last_seen_block)"
)

def load_csv(cursor, csv_file, load_sql):
    abs_path = os.path.abspath(csv_file).replace("\\", "/")
    cmd = load_sql.replace("{csv_file}", abs_path)
    logging.info(f"Execute SQL: {cmd[:100]}...")
    cursor.execute(cmd)

# ====== 主流程 / Main process ======
def main():
    # 批处理方式：每批处理 BATCH_SIZE 个文件，避免临时 CSV 占用过多磁盘空间 / Batch mode: BATCH_SIZE files are processed in a batch to avoid the temporary CSV occupying too much disk space
    checkpoint = load_checkpoint()
    logging.info(f"Load checkpoint: {checkpoint}")
    dat_files = get_dat_files()
    if not dat_files:
        logging.error("The blk*.dat file was not found in the data directory.")
        return
    next_file = checkpoint.get("next_file", "blk00000.dat")
    if next_file not in dat_files:
        logging.warning(f"The next file to be processed, {next_file}, does not exist, please add it to the directory manually.")
        sys.exit(1)
    start_index = dat_files.index(next_file)
    logging.info(f"Starting with the file {dat_files[start_index]}, There are {len(dat_files) - start_index} files.")

    # 分批处理 / Batch processing
    for i in range(start_index, len(dat_files), BATCH_SIZE):
        batch = dat_files[i:i+BATCH_SIZE]
        logging.info(f"Start parsing batches in parallel: {batch}")
        with Pool(processes=NUM_PROCESSES) as pool:
            results = pool.map(parse_dat_file, batch)
        results.sort(key=lambda x: x[1])

        # 顺序加载阶段：加载当前批次各文件的 CSV 数据 / Sequential load phase: Loads the CSV data of each file of the current batch
        try:
            conn = mysql.connector.connect(**DB_CONFIG, allow_local_infile=True)
            cursor = conn.cursor()
            logging.info("Successfully connected to MySQL database.")
            # 禁用外键和唯一性检查以加速加载 / Disable foreign key and uniqueness checks to speed up loading
            cursor.execute("SET foreign_key_checks = 0;")
            cursor.execute("SET unique_checks = 0;")
            cursor.execute("SET innodb_lock_wait_timeout = 300;")
        except mysql.connector.Error as err:
            logging.error(f"Error connecting to MySQL: {err}")
            return

        for idx, (csv_files, dat_filename) in enumerate(results):
            logging.info(f"Load the {dat_filename} data file into the database.")
            try:
                load_csv(cursor, csv_files["blocks"], INSERT_BLOCK_SQL)
                conn.commit()
                load_csv(cursor, csv_files["tx"], INSERT_TX_SQL)
                conn.commit()
                load_csv(cursor, csv_files["inputs"], INSERT_INPUT_SQL)
                conn.commit()
                load_csv(cursor, csv_files["outputs"], INSERT_OUTPUT_SQL)
                conn.commit()
                load_csv(cursor, csv_files["address"], INSERT_ADDRESS_SQL)
                conn.commit()
            except Exception as e:
                logging.error(f"Error loading CSV data: {e}")
                conn.rollback()
                cursor.close()
                conn.close()
                cleanup_temp_csv()
                sys.exit(1)
            for f in csv_files.values():
                if os.path.exists(f):
                    os.remove(f)
            next_file_candidate = get_next_filename(dat_filename)
            logging.info(f"The {dat_filename} data file is loaded, update checkpoint, and the next file should be: {next_file_candidate}")
            save_checkpoint(next_file_candidate, 0)
            next_file_path = os.path.join(DATA_DIR, next_file_candidate)
            # 如果下一个文件存在，则提示用户是否继续 / If the next file exists, the user is prompted whether to continue
            if os.path.exists(next_file_path):
                if not prompt_continue(timeout=5):
                    logging.info("User chooses to terminate program, cleaning temporary file...")
                    cleanup_temp_csv()
                    sys.exit(0)
            else:
                logging.warning(f"The next file {next_file_candidate} does not exist, please add it to the directory manually.")
                break

        cursor.execute("SET foreign_key_checks = 1;")
        cursor.execute("SET unique_checks = 1;")
        cursor.close()
        conn.close()
    logging.info("All data import is complete.")

if __name__ == "__main__":
    main()
