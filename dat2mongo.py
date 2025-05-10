import base58
import hashlib
import itertools as itr
import pymongo as db
from multiprocessing import Process, Value, Queue, freeze_support
from blockchain_parser.blockchain import Block, get_blocks
import time
from datetime import datetime
from struct import unpack
import os
import glob
import threading
import sys

# 定义创世区块前置哈希（全零字符串） / Define Genesis block prefix hash (all-zero string)
GENESIS_HASH = "0000000000000000000000000000000000000000000000000000000000000000"

# checkpoint 相关函数，保存格式为 "file_index,last_hash,block_height,utxo_size" / checkpoint related functions, saved as "file_index,last_hash,block_height,utxo_size"
def read_checkpoint():
    try:
        with open("checkpoint.txt", "r") as f:
            content = f.read().strip()
            parts = content.split(",")
            if len(parts) == 4:
                return int(parts[0]), parts[1], int(parts[2]), int(parts[3])
    except Exception:
        pass
    return 0, GENESIS_HASH, 0, 0

def update_checkpoint(file_index, curr_hash, block_height, utxo_size):
    with open("checkpoint.txt", "w") as f:
        f.write(f"{file_index},{curr_hash},{block_height},{utxo_size}")

# 改进版超时输入函数：在 Windows 下使用 msvcrt 模块检测键盘输入，其他平台采用线程方式 / Improved version of timeout input function: use msvcrt module to detect keyboard input in Windows, other platforms use thread mode
def get_input_timeout(prompt, timeout):
    if os.name == 'nt':
        import msvcrt
        sys.stdout.write(prompt)
        sys.stdout.flush()
        end_time = time.time() + timeout
        input_str = ""
        while time.time() < end_time:
            if msvcrt.kbhit():
                ch = msvcrt.getche()
                if ch in [b'\r', b'\n']:
                    break
                elif ch == b'\x08':
                    input_str = input_str[:-1]
                    sys.stdout.write("\b \b")
                else:
                    try:
                        input_str += ch.decode()
                    except UnicodeDecodeError:
                        pass
            time.sleep(0.1)
        sys.stdout.write("\n")
        return input_str
    else:
        result = [None]
        def ask():
            try:
                result[0] = input(prompt)
            except Exception:
                result[0] = None
        thread = threading.Thread(target=ask)
        thread.daemon = True
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            return None
        return result[0]

# 动态统计数据目录下所有 .dat 文件数量 / The number of.dat files in the dynamic statistics directory
file_dir = "E:/Prof Marco FYP Blockchain/data/blocks"
file_pattern = os.path.join(file_dir, "blk*.dat")
files = glob.glob(file_pattern)
files.sort()
MAX_FILE = len(files)
file_path = os.path.join(file_dir, "blk%05d.dat")

# 读取 checkpoint，如果 checkpoint 中 file_index 大于等于 MAX_FILE 则退出 / Read the checkpoint and exit if file_index in the checkpoint is greater than or equal to MAX_FILE
file_index, last_hash, saved_height, saved_utxo = read_checkpoint()
if file_index >= MAX_FILE:
    print("There are no new .dat files to process. Current checkpoint: ", file_index, ". Total files:", MAX_FILE)
    sys.exit(0)

# 地址类型常量 / Address type constant
ADDR_UNKNOWN        = int(-1)
ADDR_P2PK           = int(1)      # pay to public key
ADDR_P2PK_comp      = int(2)      # pay to compressed public key
ADDR_MULTISIG       = int(3)      # multisignature
ADDR_P2PKH          = int(10)     # pay to public key hash
ADDR_P2SH           = int(20)     # pay to script hash
ADDR_P2SH_PK        = int(21)     # pay to script hash (P2PK)
ADDR_P2SH_MULTISIG  = int(22)     # pay to script hash (multisig)
ADDR_P2SH_OTHER     = int(23)     # pay to script hash (other)
ADDR_OPRETURN       = int(0)      # op_return

def type2string(tp):
    if tp == ADDR_P2PK:
        return 'P2PK'
    elif tp == ADDR_P2PK_comp:
        return 'P2PK comp'
    elif tp == ADDR_MULTISIG:
        return 'P2PK multisig'
    elif tp == ADDR_P2PKH:
        return 'P2PKH'
    elif tp == ADDR_P2SH:
        return 'P2SH'
    elif tp == ADDR_P2SH_PK:
        return 'P2SH pubkey'
    elif tp == ADDR_P2SH_MULTISIG:
        return 'P2SH multisig'
    elif tp == ADDR_P2SH_OTHER:
        return 'P2SH unknown'
    elif tp == ADDR_OPRETURN:
        return 'OPRET'
    else:
        return 'unknown'

def keyhash2address(keyhash):
    checksum = hashlib.sha256(hashlib.sha256(b'\x00' + keyhash).digest()).digest()
    return base58.b58encode(b'\x00' + keyhash + checksum[:4])

# 计算 utxo 哈希，保持原有逻辑 / Compute the utxo hash and keep the original logic
def calcutxohash(itx, txhash):
    hash_val = int.from_bytes(txhash[:8], byteorder='big', signed=False)
    return (hash_val >> 1) ^ itx

# 解释锁定脚本，注意 P2SH 分支调用内部的 calcshorthash / Interpret the lock script and notice the calcshorthash inside the P2SH branch call
def interpret_lock_script(lock):
    def calcshorthash(keyhash):
        return int.from_bytes(keyhash[:8], byteorder='big', signed=False) >> 1
    def addr_from_keyhash(keyhash, prefix):
        checksum = hashlib.sha256(hashlib.sha256(prefix + keyhash).digest()).digest()
        return base58.b58encode(prefix + keyhash + checksum[:4])
    if ((len(lock) == 67 and lock.hex()[0:2] == '41') or (len(lock) == 35 and lock.hex()[:2] == '21')) and lock.hex()[-2:] == 'ac':
        key = (lock[1:-1]).hex()
        keyhash = hashlib.new('ripemd160', hashlib.sha256(lock[1:-1]).digest()).digest()
        tp = ADDR_P2PK if len(lock)==67 else ADDR_P2PK_comp
        return (tp, calcshorthash(keyhash), addr_from_keyhash(keyhash, b'\x00'), key)
    elif len(lock) == 25 and lock.hex()[:6] == '76a914' and lock.hex()[-4:] == '88ac':
        keyhash = lock[3:-2]
        return (ADDR_P2PKH, calcshorthash(keyhash), addr_from_keyhash(keyhash, b'\x00'), None)
    elif len(lock) == 23 and lock.hex()[:4] == 'a914' and lock.hex()[-2:] == '87':
        keyhash = lock[2:-1]
        return (ADDR_P2SH, calcshorthash(keyhash), addr_from_keyhash(keyhash, b'\x05'), None)
    elif len(lock) > 1 and lock.hex()[:2] == '6a':
        return (ADDR_OPRETURN, None, None, None)
    elif len(lock) >= 36 and lock.hex()[-2:] == 'ae':
        keyhash = hashlib.new('ripemd160', hashlib.sha256(lock).digest()).digest()
        keys = []
        hexkeys = lock[1:-2]
        for i in range(0, lock[-2] - 80):
            print(hexkeys.hex())
            l = hexkeys[0]
            if l in [33,65] and len(hexkeys) > l:
                keys.append(hexkeys[1:l+1].hex())
                hexkeys = hexkeys[l+1:]
            else:
                return (ADDR_UNKNOWN, None, None, None)
        if len(hexkeys) == 0:
            return (ADDR_MULTISIG, calcshorthash(keyhash), addr_from_keyhash(keyhash, b'\x05'), keys)
        else:
            return (ADDR_UNKNOWN, None, None, None)
    else:
        return (ADDR_UNKNOWN, None, None, None)

def interpret_unlock_script(tp, unlock):
    if tp == ADDR_P2PKH:
        return (tp, unlock[2+unlock[0]:].hex())
    elif tp == ADDR_P2SH:
        lock = []
        while len(unlock) > 0:
            if unlock[0] == 0 or unlock[0] > 78:
                unlock = unlock[1:]
            elif unlock[0] == 76:
                l = int.from_bytes(unlock[1:2], byteorder='little')
                lock = unlock[2:l+2]
                unlock = unlock[l+2:]
            elif unlock[0] == 77:
                l = int.from_bytes(unlock[1:3], byteorder='little')
                lock = unlock[3:l+3]
                unlock = unlock[l+3:]
            elif unlock[0] == 78:
                l = int.from_bytes(unlock[1:5], byteorder='little')
                lock = unlock[5:l+5]
                unlock = unlock[l+5:]
            else:
                lock = unlock[1:unlock[0]+1]
                unlock = unlock[unlock[0]+1:]
        if len(lock) > 0:
            (tp2, shorthash, addr, key) = interpret_lock_script(lock)
            if tp2 == ADDR_P2PK:
                return (ADDR_P2SH_PK, key)
            elif tp2 == ADDR_MULTISIG:
                return (ADDR_P2SH_MULTISIG, key)
            else:
                return (ADDR_P2SH_OTHER, None)
        else:
            return (ADDR_P2SH_OTHER, None)
    else:
        return (tp, None)

# 修改后的 READER PROCESS / Modified READER PROCESS
# 参数 start_hash 用于恢复上次运行时的链末尾，另外传入共享变量 h 与 utxolen 以便在用户选择停止时更新 checkpoint / The start_hash parameter is used to restore the end of the chain from the last run, and the shared variables h and utxolen are passed to update the checkpoint when the user chooses to stop
def reader_process(qread_unpack_l, qread_sync, f, b, fmax, start_hash, h, utxolen):
    curr_hash = start_hash  # 从 checkpoint 恢复当前链尾 / Restores the current chain tail from the checkpoint
    blk_pool = {}
    if f.value < fmax:
        rdr = iter(get_blocks(file_path % f.value))
        f.value += 1
    else:
        for q in qread_unpack_l:
            q.put(None)
        qread_sync.put(None)
        return
    local_h = 0
    while True:
        if curr_hash in blk_pool:
            blk = blk_pool.pop(curr_hash)
            curr_hash = blk.hash
            qread_unpack_l[local_h % 3].put(blk)
            local_h += 1
            qread_sync.put(blk.header.timestamp)
        else:
            try:
                blk = Block(next(rdr))
                blk_pool[blk.header.previous_block_hash] = blk
                b.value += 1
            except StopIteration:
                # 文件处理完毕，先更新 checkpoint——若用户选择停止，则不推进文件编号 / When the file is processed, the checkpoint is updated first - if the user chooses to stop, the file number is not advanced
                user_input = get_input_timeout("继续处理下一个文件? (Y/n): ", 10)
                if user_input is None or user_input.strip() == "" or user_input.strip().lower() == "y":
                    if f.value < fmax:
                        update_checkpoint(f.value, curr_hash, h.value, utxolen.value)
                        rdr = iter(get_blocks(file_path % f.value))
                        f.value += 1
                    else:
                        for q in qread_unpack_l:
                            q.put(None)
                        qread_sync.put(None)
                        return
                else:
                    # 用户选择停止，则保持当前文件编号（即 f.value-1），并保存当前 h 与 utxolen / If the user chooses to stop, the current file number (that is, f.vale-1) is kept and the current h and utxolen are saved
                    update_checkpoint(f.value - 1, curr_hash, h.value, utxolen.value)
                    for q in qread_unpack_l:
                        q.put(None)
                    qread_sync.put(None)
                    return

def unpacker_process(qread_unpack, qunpack_in, qunpack_hash):
    while True:
        blk = qread_unpack.get()
        if not blk:
            qunpack_in.put(None)
            qunpack_hash.put(None)
            return
        trans = blk.transactions
        qunpack_in.put(trans)
        qunpack_hash.put(trans)

def hasher_process(qunpack_hash_l, qhash_out):
    def get_hash(trans):
        if trans.is_segwit:
            txid = trans.hex[:4] + trans.hex[6:trans._offset_before_tx_witnesses] + trans.hex[-4:]
        else:
            txid = trans.hex
        return hashlib.sha256(hashlib.sha256(txid).digest()).digest()
    for h in itr.count(0):
        trans = qunpack_hash_l[h % 3].get()
        if not trans:
            qhash_out.put(None)
            return
        hash_trans = [(get_hash(t), t) for t in trans]
        qhash_out.put(hash_trans)

def dist_in_process(qunpack_in_l, qin_child_l):
    for h in itr.count(0):
        trans = qunpack_in_l[h % 3].get()
        if not trans:
            return
        tins_l = [[] for idx in range(8)]
        for t in trans:
            for tin in t.inputs:
                if tin.transaction_index != 0xffffffff:
                    utxohash = calcutxohash(tin.transaction_index, tin.hex[:32])
                    tins_l[utxohash % 8].append((tin, utxohash))
        for idx in range(8):
            qin_child_l[idx].put(tins_l[idx])

def dist_out_process(qhash_out, qout_child_l):
    for h in itr.count(0):
        trans = qhash_out.get()
        if not trans:
            for qchild in iter(qout_child_l):
                qchild.put(False)
            return
        touts_l = [[] for idx in range(8)]
        for (txhash, t) in trans:
            for itout, tout in enumerate(t.outputs):
                utxohash = calcutxohash(itout, txhash)
                touts_l[utxohash % 8].append((tout, utxohash))
        for idx in range(8):
            qout_child_l[idx].put(True)
            qout_child_l[idx].put(touts_l[idx])

def analysis_process_child(qin_child, qout_child, qchild_sync, qchild_write):
    utxo_pool = {}
    for h in itr.count(0):
        flag = qout_child.get()
        if not flag:
            return
        touts = qout_child.get()
        reqs = []
        Dtotal = 0
        Dlost = 0
        Dopret = 0
        for (tout, utxohash) in touts:
            (tp, keyhash, addr, pubkey) = interpret_lock_script(tout.script.hex)
            if tp != ADDR_OPRETURN:
                Dtotal += tout.value
                if keyhash:
                    cmd = {'$setOnInsert': {'type': tp, 'addr': addr},
                           '$inc': {'val': tout.value, 'key-seen': int(pubkey is not None)}}
                    reqs.append(db.UpdateOne({'_id': keyhash}, cmd, upsert=True))
                else:
                    Dlost += tout.value
            else:
                Dopret += tout.value
            utxo_pool[utxohash] = (tp, tout.value, keyhash)
        tins = qin_child.get()
        for (tin, utxohash) in tins:
            try:
                (tp, val, keyhash) = utxo_pool.pop(utxohash)
            except KeyError:
                continue
            Dtotal -= val
            if keyhash:
                (tp, pubkey) = interpret_unlock_script(tp, tin.script.hex)
                cmd = {'$set': {'type': tp}, '$inc': {'val': -val, 'key-seen': int(pubkey is not None)}}
                reqs.append(db.UpdateOne({'_id': keyhash}, cmd))
            else:
                Dlost -= val
        qchild_write.put(reqs)
        qchild_sync.put((Dtotal, Dlost, Dopret, len(utxo_pool)))

def sync_process(qchild_sync_l, qread_sync, qsync_write, h, utxolen):
    total = 0
    opret = 0
    lost = 0
    while True:
        curr_date = qread_sync.get()
        if not curr_date:
            qsync_write.put(False)
            return
        utxotot = 0
        for qchild in qchild_sync_l:
            (Dtotal, Dlost, Dopret, utxo) = qchild.get()
            total += Dtotal
            lost += Dlost
            opret += Dopret
            utxotot += utxo
        utxolen.value = utxotot
        h.value += 1
        qsync_write.put(True)
        qsync_write.put((curr_date, total, lost, opret, h.value))

def writer_process(qchild_write_l, qsync_write):
    FLUSH_DATABASE = 1
    MIN_FLUSH = 1000
    PRINT_STATUS = 150
    TAKE_SNAPSHOT = 3000
    client = db.MongoClient()
    lookup = client['btc_2']['addr']
    snapshot = client['btc_2']['snap']
    def take_snapshot(date, total, lost, opret, height):
        snapshot.insert_one({'_id': height, 'date': date, 'tot-val': total, 'op-return': opret, 'unknown': lost})
        tpsummary = lookup.aggregate([
            {'$match': {'val': {'$gt': 0}}},
            {'$group': {'_id': '$type', 'num-pos': {'$sum': 1}, 'tot-val': {'$sum': '$val'}}}
        ])
        for doc in tpsummary:
            doc['type'] = type2string(doc.pop('_id'))
            snapshot.update_one({'_id': height}, {'$push': {'summary-by-type': doc}})
        qsummary = lookup.aggregate([
            {'$match': {'key-seen': {'$gt': 0}, 'val': {'$gt': 0}}},
            {'$group': {'_id': '$type', 'num-pos': {'$sum': 1}, 'tot-val': {'$sum': '$val'}}}
        ])
        qval = 0
        for doc in qsummary:
            doc['type'] = type2string(doc.pop('_id'))
            qval += doc['tot-val']
            snapshot.update_one({'_id': height}, {'$push': {'quantum-by-type': doc}})
        snapshot.update_one({'_id': height}, {'$set': {'qattack-frac': float(qval) / float(total), 'unknown-frac': float(lost) / float(total)}})
    reqs = []
    while True:
        flag = qsync_write.get()
        if not flag:
            if reqs:
                lookup.bulk_write(reqs, ordered=True, bypass_document_validation=True)
            take_snapshot(date, total, lost, opret, height)
            print('final database update (snapshot taken)')
            return
        (date, total, lost, opret, height) = qsync_write.get()
        for qchild in iter(qchild_write_l):
            reqs.extend(qchild.get())
        if ((height % TAKE_SNAPSHOT == 0) and reqs) or ((height % FLUSH_DATABASE == 0) and (len(reqs) > MIN_FLUSH)):
            lookup.bulk_write(reqs, ordered=True, bypass_document_validation=True)
            reqs = []
        if height % PRINT_STATUS == 0:
            status = ('database update at height: ' + str(height) +
                      '   block date: ' + str(date)[0:10] +
                      '   cirulating: %.3f kBTC' % (float(total)*1e-11) +
                      '   unknown: %02.2f%%' % (float(lost)/float(total)*100.) +
                      '   op_ret: %.3f mBTC' % (float(opret)*1e-5))
            if height % TAKE_SNAPSHOT == 0:
                take_snapshot(date, total, lost, opret, height)
                status += '  (snapshot taken)'
            print(status + '\n', end='', flush=True)

if __name__ == '__main__':
    freeze_support()
    # 读取 checkpoint后初始化共享变量（文件号、区块高度、utxo 数量） / Initialize shared variables (file number, block height, number of UTXOs) after reading checkpoint
    file_index, last_hash, saved_height, saved_utxo = read_checkpoint()
    qread_unpack_l = [Queue(12) for i in range(3)]
    qread_sync = Queue(36)
    qunpack_hash_l = [Queue(6) for i in range(3)]
    qunpack_in_l = [Queue(6) for i in range(3)]
    qhash_out = Queue(6)
    qout_child_l = [Queue(6*2) for i in range(8)]
    qin_child_l = [Queue(6) for i in range(8)]
    qchild_sync_l = [Queue(6) for i in range(8)]
    qchild_write_l = [Queue(36) for i in range(8)]
    qsync_write = Queue(36*2)

    f = Value('i', file_index)
    b = Value('i', 0)
    h = Value('i', saved_height)
    utxolen = Value('i', saved_utxo)

    p_read = Process(target=reader_process, args=(qread_unpack_l, qread_sync, f, b, MAX_FILE, last_hash, h, utxolen))
    p_unpack_l = [Process(target=unpacker_process, args=(qread_unpack_l[i], qunpack_in_l[i], qunpack_hash_l[i])) for i in range(3)]
    p_hash = Process(target=hasher_process, args=(qunpack_hash_l, qhash_out))
    p_dist_in = Process(target=dist_in_process, args=(qunpack_in_l, qin_child_l))
    p_dist_out = Process(target=dist_out_process, args=(qhash_out, qout_child_l))
    p_analyse_child_l = [Process(target=analysis_process_child, args=(qin_child_l[i], qout_child_l[i], qchild_sync_l[i], qchild_write_l[i])) for i in range(8)]
    p_sync = Process(target=sync_process, args=(qchild_sync_l, qread_sync, qsync_write, h, utxolen))
    p_write = Process(target=writer_process, args=(qchild_write_l, qsync_write))

    T = datetime.now()

    p_read.start()
    for process in p_unpack_l:
        process.start()
    p_hash.start()
    p_dist_in.start()
    p_dist_out.start()
    for process in p_analyse_child_l:
        process.start()
    p_sync.start()
    p_write.start()

    while p_write.is_alive():
        time.sleep(1)
        delta = datetime.now() - T
        print('running time: %02d:%02d:%02d' % (delta.seconds // 3600 + delta.days * 24,
                                                (delta.seconds // 60) % 60, delta.seconds % 60) +
              '   input file: ' + str(f.value) + '/' + str(MAX_FILE) +
              '   height: ' + str(h.value) +
              '   utxo size: ' + str(utxolen.value) + '\n', end='', flush=True)
