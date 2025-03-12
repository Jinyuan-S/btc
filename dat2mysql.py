#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dat2MySQL.py

完整解析 Bitcoin dat 文件，将区块、交易、交易输入、交易输出、地址信息写入 MySQL 数据库。
要求：
  1. 解析 dat 文件内所有信息，不遗漏。
  2. 支持断点续传（记录下一个要处理的文件名及该文件内的偏移）。
  3. 利用批量写入加速数据导入。
  4. 自动衔接新 dat 文件。
  5. 当一个文件处理完毕且无错误后，checkpoint 中的文件名自动 +1（格式保持一致）且 offset 重置为 0；
     如果下一个文件不存在，则提示用户需要手动添加该文件，然后结束程序。
"""

import os
import json
import struct
import logging
import hashlib
import mysql.connector
from mysql.connector import errorcode
from io import BytesIO
import sys

# ====== 配置区 ======
DATA_DIR = "./dat"          # 存放 blk*.dat 文件的目录
CHECKPOINT_FILE = "checkpoint.json"
# 批量插入阈值（可根据实际情况调整）
BATCH_SIZE_BLOCKS = 5
BATCH_SIZE_TX = 500
BATCH_SIZE_INPUT = 500
BATCH_SIZE_OUTPUT = 500
BATCH_SIZE_ADDRESS = 100

# MySQL 连接配置
DB_CONFIG = {
    "user": "root",
    "password": "gsj123",
    "host": "localhost",
    "port": 3306,
    "database": "btc_analysis",
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# ====== Base58 与 Bech32 辅助函数 ======
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

# Bech32 实现（参考 BIP-0173）
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

# ====== 定义 read_varint ======
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
    else:  # 0xff
        data = stream.read(8)
        return int.from_bytes(data, 'little')

# ====== 计算双 SHA256 ======
def double_sha256(data: bytes) -> bytes:
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()

# ====== 地址提取函数 ======
def extract_address(script_hex: str):
    # P2PKH: 76a914{20-byte-hash}88ac (50 hex字符)
    if script_hex.startswith("76a914") and script_hex.endswith("88ac") and len(script_hex) == 50:
        hash160 = bytes.fromhex(script_hex[6:46])
        addr = base58_check_encode(b'\x00' + hash160)
        return addr, "P2PKH"
    # P2SH: a914{20-byte-hash}87 (46 hex字符)
    elif script_hex.startswith("a914") and script_hex.endswith("87") and len(script_hex) == 46:
        hash160 = bytes.fromhex(script_hex[4:44])
        addr = base58_check_encode(b'\x05' + hash160)
        return addr, "P2SH"
    # P2WPKH: 0014{20-byte-hash} (44 hex字符)
    elif script_hex.startswith("0014") and len(script_hex) == 44:
        hash160 = bytes.fromhex(script_hex[4:44])
        addr = bech32_encode("bc", 0, list(hash160))
        return addr, "P2WPKH"
    # P2WSH: 0020{32-byte-hash} (68 hex字符)
    elif script_hex.startswith("0020") and len(script_hex) == 68:
        hash256 = bytes.fromhex(script_hex[4:68])
        addr = bech32_encode("bc", 0, list(hash256))
        return addr, "P2WSH"
    else:
        return None, "unknown"

# ====== 解析交易、区块 ======
def parse_transaction(stream: BytesIO) -> (dict, bytes):
    """
    解析交易：
      version (4字节) | tx_in_count (varint) | 每个输入 | tx_out_count (varint) | 每个输出 | lock_time (4字节)
    返回交易数据及原始字节
    """
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
        logging.error(f"解析交易时出错：{e}")
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
        "output_count": out_count,
    }
    return tx, tx_raw

def parse_block(file_obj, file_name: str) -> dict:
    """
    解析一个区块：
      先读取8字节 (magic + length)，再读取整个区块数据。
    返回字典包含：
      block_hash, version, prev_block_hash, merkle_root, timestamp, bits, nonce, block_size, tx_count, raw_block, transactions,
      以及数据来源 file_name 和 file_offset。
    """
    start_offset = file_obj.tell()
    header = file_obj.read(8)
    if len(header) < 8:
        return None
    magic = header[:4]
    if magic != b'\xf9\xbe\xb4\xd9':
        logging.error(f"无效的 magic {magic.hex()} 在 {file_name} offset {start_offset}")
        return None
    block_length = int.from_bytes(header[4:8], 'little')
    block_data = file_obj.read(block_length)
    if len(block_data) != block_length:
        logging.info(f"区块数据不完整，在 {file_name} offset {start_offset}，可能已到文件末尾。")
        return None
    raw_block = header + block_data
    block_size = len(raw_block)

    if len(block_data) < 80:
        logging.info(f"区块数据太短，无法解析区块头，在 {file_name} offset {start_offset}。")
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
        logging.error(f"读取交易数量失败：{e}")
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
            logging.error(f"解析第 {i} 笔交易失败：{e}")
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

# ====== 断点续传相关 ======
def load_checkpoint() -> dict:
    """
    加载 checkpoint。如果不存在或为空，则返回初始值，
    表示从 "blk00000.dat" 开始、offset 为 0。
    """
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r") as f:
                content = f.read().strip()
                if not content:
                    return {"next_file": "blk00000.dat", "offset": 0}
                return json.loads(content)
        except Exception as e:
            logging.error(f"读取 checkpoint 失败: {e}")
            return {"next_file": "blk00000.dat", "offset": 0}
    else:
        return {"next_file": "blk00000.dat", "offset": 0}

def save_checkpoint(next_file: str, offset: int):
    """
    保存 checkpoint，其中 next_file 为下一个要处理的 dat 文件，
    offset 为该文件中的偏移位置。
    """
    checkpoint = {"next_file": next_file, "offset": offset}
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f)

def get_dat_files() -> list:
    files = [f for f in os.listdir(DATA_DIR) if f.startswith("blk") and f.endswith(".dat")]
    files.sort()
    return files

def get_next_filename(current_file: str) -> str:
    """
    根据当前文件名计算下一个文件名，假设格式为 "blkNNNNN.dat"。
    """
    try:
        num = int(current_file[3:8])
    except Exception:
        num = 0
    next_num = num + 1
    return f"blk{next_num:05d}.dat"

# ====== SQL 语句 ======
INSERT_BLOCK_SQL = (
    "INSERT IGNORE INTO blocks "
    "(block_hash, version, prev_block_hash, merkle_root, timestamp, bits, nonce, block_size, tx_count, raw_block, file_name, file_offset) "
    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
)
INSERT_TX_SQL = (
    "INSERT IGNORE INTO transactions "
    "(txid, block_hash, version, input_count, output_count, lock_time, raw_tx) "
    "VALUES (%s, %s, %s, %s, %s, %s, %s)"
)
INSERT_INPUT_SQL = (
    "INSERT IGNORE INTO tx_inputs "
    "(txid, input_index, prev_txid, prev_output_index, script_sig, sequence) "
    "VALUES (%s, %s, %s, %s, %s, %s)"
)
INSERT_OUTPUT_SQL = (
    "INSERT IGNORE INTO tx_outputs "
    "(txid, output_index, value, script_pub_key) "
    "VALUES (%s, %s, %s, %s)"
)
INSERT_ADDRESS_SQL = (
    "INSERT INTO address (address, address_type, total_received, total_sent, balance, pubkey_revealed, first_seen_block, last_seen_block) "
    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
    "ON DUPLICATE KEY UPDATE "
    "total_received = total_received + VALUES(total_received), "
    "total_sent = total_sent + VALUES(total_sent), "
    "balance = (total_received + VALUES(total_received)) - (total_sent + VALUES(total_sent)), "
    "last_seen_block = VALUES(last_seen_block), "
    "pubkey_revealed = GREATEST(pubkey_revealed, VALUES(pubkey_revealed))"
)

# ====== 全局数据结构 ======
utxo = {}          # key: (txid, output_index) -> (address, value)
address_updates = {}  # key: address -> 更新数据字典

def update_address(addr, addr_type, received=0, sent=0, block_hash=None, pubkey_flag=0):
    if addr is None:
        return
    if addr not in address_updates:
        address_updates[addr] = {
            "address": addr,
            "address_type": addr_type if addr_type else "unknown",
            "total_received": received,
            "total_sent": sent,
            "balance": received - sent,
            "pubkey_revealed": pubkey_flag,
            "first_seen_block": block_hash,
            "last_seen_block": block_hash,
        }
    else:
        rec = address_updates[addr]
        rec["total_received"] += received
        rec["total_sent"] += sent
        rec["balance"] = rec["total_received"] - rec["total_sent"]
        rec["last_seen_block"] = block_hash
        if pubkey_flag:
            rec["pubkey_revealed"] = 1

def flush_address_updates(cursor):
    global address_updates
    if not address_updates:
        return
    batch = []
    for rec in address_updates.values():
        batch.append((
            rec["address"],
            rec["address_type"],
            rec["total_received"],
            rec["total_sent"],
            rec["balance"],
            rec["pubkey_revealed"],
            rec["first_seen_block"],
            rec["last_seen_block"],
        ))
    try:
        cursor.executemany(INSERT_ADDRESS_SQL, batch)
    except Exception as e:
        logging.error(f"地址批量更新错误: {e}")
    address_updates = {}

# ====== 主处理逻辑 ======
def main():
    checkpoint = load_checkpoint()
    logging.info(f"加载 checkpoint: {checkpoint}")
    dat_files = get_dat_files()
    if not dat_files:
        logging.error("数据目录中未发现 blk*.dat 文件。")
        return

    # 如果 checkpoint 中指定的 next_file 不存在于目录中，则提示并退出
    next_file = checkpoint.get("next_file", "blk00000.dat")
    if next_file not in dat_files:
        logging.warning(f"下一个要处理的文件 {next_file} 不存在，请手动添加到目录中。")
        sys.exit(1)

    start_index = dat_files.index(next_file)
    total_files = len(dat_files)

    try:
        cnx = mysql.connector.connect(**DB_CONFIG)
        cursor = cnx.cursor()
        logging.info("成功连接到 MySQL 数据库。")
    except mysql.connector.Error as err:
        logging.error(f"连接 MySQL 出错: {err}")
        return

    block_batch = []
    tx_batch = []
    input_batch = []
    output_batch = []

    def flush_batches():
        nonlocal block_batch, tx_batch, input_batch, output_batch
        try:
            if block_batch:
                cursor.executemany(INSERT_BLOCK_SQL, block_batch)
                block_batch = []
            if tx_batch:
                cursor.executemany(INSERT_TX_SQL, tx_batch)
                tx_batch = []
            if input_batch:
                cursor.executemany(INSERT_INPUT_SQL, input_batch)
                input_batch = []
            if output_batch:
                cursor.executemany(INSERT_OUTPUT_SQL, output_batch)
                output_batch = []
            flush_address_updates(cursor)
            cnx.commit()
        except mysql.connector.Error as err:
            logging.error(f"批量插入错误: {err}")
            cnx.rollback()

    for idx in range(start_index, total_files):
        current_file = dat_files[idx]
        logging.info(f"处理文件 {idx+1}/{total_files}: {current_file}")
        file_path = os.path.join(DATA_DIR, current_file)
        with open(file_path, "rb") as file_obj:
            if idx == start_index:
                file_obj.seek(checkpoint["offset"])
            else:
                file_obj.seek(0)
            while True:
                cur_offset = file_obj.tell()
                block = parse_block(file_obj, current_file)
                if block is None:
                    break
                block_tuple = (
                    block["block_hash"],
                    block["version"],
                    block["prev_block_hash"],
                    block["merkle_root"],
                    block["timestamp"],
                    block["bits"],
                    block["nonce"],
                    block["block_size"],
                    block["tx_count"],
                    block["raw_block"],
                    block["file_name"],
                    block["file_offset"],
                )
                block_batch.append(block_tuple)
                for tx in block["transactions"]:
                    tx_tuple = (
                        tx["txid"],
                        block["block_hash"],
                        tx["version"],
                        tx["input_count"],
                        tx["output_count"],
                        tx["lock_time"],
                        tx["raw_tx"],
                    )
                    tx_batch.append(tx_tuple)
                    for idx_in, txin in enumerate(tx["inputs"]):
                        input_tuple = (
                            tx["txid"],
                            idx_in,
                            txin["prev_txid"],
                            txin["prev_index"],
                            txin["script_sig"],
                            txin["sequence"],
                        )
                        input_batch.append(input_tuple)
                        if txin["prev_txid"] != "0"*64:
                            key = (txin["prev_txid"], txin["prev_index"])
                            if key in utxo:
                                addr, value = utxo.pop(key)
                                update_address(addr, None, sent=value, block_hash=block["block_hash"])
                    for idx_out, txout in enumerate(tx["outputs"]):
                        output_tuple = (
                            tx["txid"],
                            idx_out,
                            txout["value"],
                            txout["script_pub_key"],
                        )
                        output_batch.append(output_tuple)
                        addr, addr_type = extract_address(txout["script_pub_key"])
                        if addr:
                            update_address(addr, addr_type, received=txout["value"], block_hash=block["block_hash"])
                            utxo[(tx["txid"], idx_out)] = (addr, txout["value"])
                if len(block_batch) >= BATCH_SIZE_BLOCKS:
                    flush_batches()
                    save_checkpoint(current_file, file_obj.tell())
                if len(tx_batch) >= BATCH_SIZE_TX or len(input_batch) >= BATCH_SIZE_INPUT or len(output_batch) >= BATCH_SIZE_OUTPUT:
                    flush_batches()
                save_checkpoint(current_file, file_obj.tell())
        logging.info(f"文件 {current_file} 处理完毕。")
        # 文件处理完毕后，计算下一个文件名，并更新 checkpoint
        next_file = get_next_filename(current_file)
        next_file_path = os.path.join(DATA_DIR, next_file)
        if not os.path.exists(next_file_path):
            logging.warning(f"下一个文件 {next_file} 不存在，请手动添加到目录中。")
            save_checkpoint(next_file, 0)
            break
        else:
            save_checkpoint(next_file, 0)
    flush_batches()
    # 不再重置 checkpoint，保持最新 checkpoint
    cursor.close()
    cnx.close()
    logging.info("全部数据导入完成。")

if __name__ == "__main__":
    main()
