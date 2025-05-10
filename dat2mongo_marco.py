"""
    This is Marco's initial code. No significant modifications have been made. Only some comments have been added.
"""

import base58
import hashlib
import itertools as itr
import pymongo as db
from multiprocessing import Process, Value, Queue
from blockchain_parser.blockchain import Block, get_blocks
import time
from datetime import datetime
from struct import unpack

# /dat/blk00000.dat

# maximum file number (ignore last file because of possible orphans) / 最大文件数（忽略最后一个文件，因为可能有孤儿）
MAX_FILE = 1234
file_path = 'E:/Prof Marco FYP Blockchain/data/blocks/blk%05d.dat'
NUM_UNPACKERS = 3  # 4 cores
NUM_ANALYZERS = 8  # 12 cores (plus reader, hasher, etc.)


# flags for address types / 地址类型标志
ADDR_UNKNOWN        = int(-1)
ADDR_P2PK           = int(1)      # pay to public key / 向公钥支付
ADDR_P2PK_comp      = int(2)      # pay to compressed public key / 支付压缩公钥
ADDR_MULTISIG       = int(3)      # multisignature (any variant) / 多重签名（任何变体）
ADDR_P2PKH          = int(10)     # pay to public key hash / 支付给公钥哈希
ADDR_P2SH           = int(20)     # pay to script hash (not yet known - unspent) / 支付到脚本散列（尚不知道-未花费）
ADDR_P2SH_PK        = int(21)     # pay to script hash (P2PK script) / 支付到脚本散列（P2PK脚本）
ADDR_P2SH_MULTISIG  = int(22)     # pay to script hash (multisig script) / 支付脚本散列（multisig脚本）
ADDR_P2SH_OTHER     = int(23)     # pay to script hash (other script) / 支付到脚本哈希（其他脚本）
ADDR_OPRETURN       = int(0)      # op_return - unspendable / Op_return -不可花费

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

# calcualte address from key hash / 从键哈希计算地址
def keyhash2address(keyhash):
    checksum = hashlib.sha256(hashlib.sha256(b'\x00' + keyhash).digest()).digest()
    return base58.b58encode(b'\x00' + keyhash + checksum[:4])

# transform index + transaction hash into a single shorter hash / 将索引+事务哈希转换为单个更短的哈希
def calcutxohash(itx, txhash):
    # represent first 8 bytes as long python integer / 将前8个字节表示为长python整数
    # this number could be increased to a maximum of 32 (full length) to completely avoid collisions / 这个数字可以增加到最大32（全长），以完全避免碰撞
    hash = int.from_bytes(txhash[:8], byteorder='big', signed=False)
    return (hash >> 1) ^ itx


def interpret_lock_script(lock):
    # returns short keyhash, it will be used for the database so let us hope it is unique / 返回短keyhash，它将用于数据库，所以让我们希望它是唯一的
    def calcshorthash(keyhash):
        return int.from_bytes(keyhash[:8], byteorder='big', signed=False) >> 1

    # calculate address (stored as string) from keyhash / 从keyhash中计算地址（存储为字符串）
    def addr_from_keyhash(keyhash, prefix):
        checksum = hashlib.sha256(hashlib.sha256(prefix + keyhash).digest()).digest()
        return base58.b58encode(prefix + keyhash + checksum[:4])

    # P2PK
    #
    # 41 04678afdb0fe5548271967f1a67130b7105cd6a828e03909a67962e0ea1f61deb649f6bc3f4cef38c4f35504e51ec112de5c384df7ba0b8d578a4c702b6bf11d5f ac
    #
    # PUSHDATA(65) 04678afdb0fe5548271967f1a67130b7105cd6a828e03909a67962e0ea1f61deb649f6bc3f4cef38c4f35504e51ec112de5c384df7ba0b8d578a4c702b6bf11d5f OP_CHECKSIG
    if ((len(lock) == 67 and lock.hex()[0:2] == '41') or (len(lock) == 35 and lock.hex()[:2] == '21')) and lock.hex()[-2:] == 'ac':
        key = (lock[1:-1]).hex()
        keyhash = hashlib.new('ripemd160', hashlib.sha256(lock[1:-1]).digest()).digest()
        if len(lock) == 67:
            tp = ADDR_P2PK
        else:
            tp = ADDR_P2PK_comp
        return (tp, calcshorthash(keyhash), addr_from_keyhash(keyhash, b'\x00'), key)

    # P2PKH
    #
    # 76 a9 14 12ab8dc588ca9d5787dde7eb29569da63c3a238c 88 ac
    #
    # OP_DUP OP_HASH160 PUSHDATA(20) 12ab8dc588ca9d5787dde7eb29569da63c3a238c OP_EQUALVERIFY OP_CHECKSIG
    elif len(lock) == 25 and lock.hex()[:6] == '76a914' and lock.hex()[-4:] == '88ac':
        keyhash = lock[3:-2]
        return (ADDR_P2PKH, calcshorthash(keyhash), addr_from_keyhash(keyhash, b'\x00'), None)

    # P2SH
    #
    # a9 14 e9c3dd0c07aac76179ebc76a6c78d4d67c6c160a 87
    #
    # OP_HASH160 PUSHDATA(20) e9c3dd0c07aac76179ebc76a6c78d4d67c6c160a OP_EQUAL
    elif len(lock) == 23 and lock.hex()[:4] == 'a914' and lock.hex()[-2:] == '87':
        keyhash = lock[2:-1]
        return (ADDR_P2SH, calcshorthash(keyhash), addr_from_keyhash(keyhash, b'\x05'), None)

    # OP_RETURN
    elif len(lock) > 1 and lock.hex()[:2] == '6a':
        return (ADDR_OPRETURN, None, None, None)

    # MULTISIG
    #
    # 52 41 04a882d414e478039cd5b52a92ffb13dd5e6bd4515497439dffd691a0f12af9575fa349b5694ed3155b136f09e63975a1700c9f4d4df849323dac06cf3bd6458cd
    # 41 046ce31db9bdd543e72fe3039a1f1c047dab87037c36a669ff90e28da1848f640de68c2fe913d363a51154a0c62d7adea1b822d05035077418267b1a1379790187
    # 41 0411ffd36c70776538d079fbae117dc38effafb33304af83ce4894589747aee1ef992f63280567f52f5ba870678b4ab4ff6c8ea600bd217870a8b4f1f09f3a8e83 53 ae
    #
    # OP_2 PUSHDATA(65) 04a882d414e478039cd5b52a92ffb13dd5e6bd4515497439dffd691a0f12af9575fa349b5694ed3155b136f09e63975a1700c9f4d4df849323dac06cf3bd6458cd
    # PUSHDATA(65) 046ce31db9bdd543e72fe3039a1f1c047dab87037c36a669ff90e28da1848f640de68c2fe913d363a51154a0c62d7adea1b822d05035077418267b1a1379790187
    # PUSHDATA(65) 0411ffd36c70776538d079fbae117dc38effafb33304af83ce4894589747aee1ef992f63280567f52f5ba870678b4ab4ff6c8ea600bd217870a8b4f1f09f3a8e83 OP_3 OP_CHECKMULTISIG
    elif len(lock) >= 36 and lock.hex()[-2:] == 'ae':
        # Address is defined to be the corresponding P2SH address / Address定义为对应的P2SH地址
        keyhash = hashlib.new('ripemd160', hashlib.sha256(lock).digest()).digest()
        keys = []
        hexkeys = lock[1:-2]
        for i in range(0,lock[-2]-80):
            # Check if a valid key / 检查是否有有效的密钥
            print(hexkeys.hex())
            l = hexkeys[0]
            if l in [33,65] and len(hexkeys)>l:
                keys.append(hexkeys[1:l+1].hex())
                hexkeys = hexkeys[l+1:]
            else:
                return (ADDR_UNKNOWN, None, None, None)
        if len(hexkeys) == 0:
            return (ADDR_MULTISIG, calcshorthash(keyhash), addr_from_keyhash(keyhash, b'\x05'), keys)
        else:
            return (ADDR_UNKNOWN, None, None, None)

    # UNKNOWN
    else:
        return (ADDR_UNKNOWN, None, None, None)

def interpret_unlock_script(tp, unlock):
    # returns public key / 返回公钥
    if tp == ADDR_P2PKH:
        return (tp, unlock[2+unlock[0]:].hex())
    elif tp == ADDR_P2SH:
        lock = []; # Take the lock script to be the last data pushed / 将锁脚本作为最后推送的数据
        while len(unlock)>0:
            # Regular opcode / 常规的操作码
            if unlock[0] == 0 or unlock[0]>78:
                unlock = unlock[1:]
            # PUSHDATA1
            elif unlock[0] == 76:
                l = int.from_bytes(unlock[1:2],byteorder='little')
                lock = unlock[2:l+2]
                unlock = unlock[l+2:]
            # PUSHDATA2
            elif unlock[0] == 77:
                l = int.from_bytes(unlock[1:3],byteorder='little')
                lock = unlock[3:l+3]
                unlock = unlock[l+3:]
            # PUSHDATA4, can't actually be used / 不能实际使用
            elif unlock[0] == 78:
                l = int.from_bytes(unlock[1:5],byteorder='little')
                lock = unlock[5:l+5]
                unlock = unlock[l+5:]
            # Other PUSHDATA
            else:
                lock = unlock[1:unlock[0]+1]
                unlock = unlock[unlock[0]+1:]

        if len(lock)>0:
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


# READER PROCESS / 读者的过程
# this process reads files, orders the blocks, and pushes them into two queues for analysis / 该进程读取文件，对块排序，并将它们推入两个队列进行分析
def reader_process(qread_unpack_l, qread_sync, f, b, fmax):
    # previous hash of genesis block is set to this / 生成块的前一个哈希值设置为此
    curr_hash = '0000000000000000000000000000000000000000000000000000000000000000'
    # block pool / 块池
    # indexed by previous hash / 按前一个散列索引
    blk_pool = {}
    # open first file / 打开第一个文件
    rdr = iter([])
    h = 0

    while True:
        # if next block is not currently in the pool, read the next file / 如果下一个块当前不在池中，则读取下一个文件
        if curr_hash in blk_pool:
            blk = blk_pool.pop(curr_hash)
            curr_hash = blk.hash
            # unpack block in next process / 在下一个进程中解包块
            qread_unpack_l[h % 3].put(blk)
            h += 1
            # submit some block info for database pushes / 为数据库推送提交一些块信息
            qread_sync.put(blk.header.timestamp)
        else:
            try:
                blk = Block(next(rdr))
                # update poool / 更新池
                blk_pool.update({blk.header.previous_block_hash: blk})
                # increment block counter / 增量块计数器
                b.value += 1
            except StopIteration:
                # time to move to the next file / 是时候转到下一个文件了
                if f.value < fmax:
                    rdr = iter(get_blocks(file_path % f.value))
                    f.value += 1
                else:
                    # we are done here / 我们做完了
                    # let the analysis process know / 让分析过程知道
                    qread_unpack_l[0].put(None)
                    qread_unpack_l[1].put(None)
                    qread_unpack_l[2].put(None)
                    qread_sync.put(None)
                    return

# UNPACKER PROCESS / 解包器过程
# does the actual unpacking of the blocks, resulting in a list of transactions / 实际解包块，产生一个交易列表吗
# this process is very slow, likely due to the implemention chosen in the module we use / 这个过程非常缓慢，可能是由于我们使用的模块中选择的实现
def unpacker_process(qread_unpack, qunpack_in, qunpack_hash):

    while True:
        blk = qread_unpack.get()
        if not blk:
            # we are done here / 我们做完了
            qunpack_in.put(None)
            qunpack_hash.put(None)
            return

        # create list of transactions / 创建事务列表
        trans = blk.transactions
        qunpack_in.put(trans)
        qunpack_hash.put(trans)

# HASHER PROCESS / 哈希的过程
# calculates all transaction hashes / 计算所有事务哈希值
def hasher_process(qunpack_hash_l, qhash_out):

    def get_hash(trans):
        if trans.is_segwit:
            txid = trans.hex[:4] + trans.hex[6:trans._offset_before_tx_witnesses] + trans.hex[-4:]
        else:
            txid = trans.hex
        # calculate SHA256(SHA256(txid)) / 计算SHA256 (SHA256 (txid))
        return hashlib.sha256(hashlib.sha256(txid).digest()).digest()

    for h in itr.count(0):
        trans = qunpack_hash_l[h % 3].get()
        if not trans:
            # we are done here / 我们做完了
            qhash_out.put(None)
            return

        # calculate transaction hash / 计算事务哈希值
        hash_trans = [ (get_hash(t), t) for t in trans ]
        # push forward to output distributor / 向前推至输出分配器
        qhash_out.put(hash_trans)


# TRANSACTION INPUT DISTRIBUTOR / 事务输入分配器
# this process takes blocks from the queue and analyses their inputs / 这个过程从队列中获取块并分析它们的输入
# maintains some status information and synchronizes the database flushes / 维护一些状态信息并同步数据库刷新
def dist_in_process(qunpack_in_l, qin_child_l):
    for h in itr.count(0):
        # get next block / 下一个块
        trans = qunpack_in_l[h % NUM_UNPACKERS].get()
        if not trans:
            # we are done here / 我们做完了
            return

        tins_l = [ [] for idx in range(8) ]

        for t in trans:
            for tin in t.inputs:
                # inputs either are coinbase, or inherit type from corresponding UTXO / 输入要么是coinbase，要么是从相应的UTXO继承类型
                if tin.transaction_index != 0xffffffff:
                    # calculate utxo hash / 计算utxo哈希值
                    utxohash = calcutxohash(tin.transaction_index, tin.hex[:32])
                    # put it into the right batch / 把它放到合适的批次里
                    tins_l[utxohash % NUM_ANALYZERS].append( (tin, utxohash) )

        # send transaction inputs to the children / 向子节点发送事务输入
        for idx in range(NUM_ANALYZERS):
            qin_child_l[idx].put(tins_l[idx])


# TRANSACTION OUTPUT DISTRIBUTOR / 事务输出分配器
# this process takes blocks from the input queue and analyses all transaction outputs / 此流程从输入队列中获取块并分析所有事务输出
# it will put all database transaction generated into qout and all additions ot the utxo pool to qsync / 它将把生成的所有数据库事务放到qout中，并把所有添加到utxo池中的事务放到qsync中
def dist_out_process(qhash_out, qout_child_l):

    for h in itr.count(0):
        # get next transaction list (corresponding to one block) / 获取下一个交易列表（对应于一个区块）
        trans = qhash_out.get()
        if not trans:
            # we are done here, let everybody know / 我们结束了，告诉大家
            for qchild in iter(qout_child_l):
                qchild.put(False)
            return

        touts_l = [ [] for idx in range(8) ]

        for (txhash, t) in trans:
            for itout, tout in enumerate(t.outputs):
                # calculate utxo hash / 计算utxo哈希值
                utxohash = calcutxohash(itout, txhash)
                # put it into the right batch / 把它放到合适的批次里
                touts_l[utxohash % 8].append( (tout, utxohash) )

        # send transaction inputs to the children / 向子节点发送事务输入
        for idx in range(8):
            qout_child_l[idx].put(True)
            qout_child_l[idx].put(touts_l[idx])


# ANALYSER PROCESS (CHILD) / 分析器进程（子进程）
# handles a fraction of the UTXO pool / 处理UTXO池的一小部分
def analysis_process_child(qin_child, qout_child, qchild_sync, qchild_write):
    # unspent transaction outputs / 未使用的事务输出
    # indexed by both transaction hash and output index / 由事务哈希和输出索引索引
    utxo_pool = {}

    for h in itr.count(0):
        # get next utxo pool update / 获取下一个utxo池更新
        if not qout_child.get():
            # we are done here / 我们做完了
            return

        # reset variables / 重置变量
        reqs = []
        Dtotal = 0
        Dlost = 0
        Dopret = 0

        # work through transaction outputs first / 首先处理事务输出
        touts = qout_child.get()

        for (tout, utxohash) in touts:
            # interpret transaction locking script / 解释事务锁定脚本
            (tp, keyhash, addr, pubkey) = interpret_lock_script(tout.script.hex)

            if tp != ADDR_OPRETURN:
                # Satoshis sent added to total / 中本聪发送到总数中
                Dtotal += tout.value

                if keyhash:
                    # check if database entry with this address already exists, and update it if found / 检查具有此地址的数据库条目是否已经存在，如果找到则更新它
                    cmd = {'$setOnInsert': {'type': tp, 'addr': addr},
                           '$inc': {'val': tout.value, 'key-seen': int(pubkey != None)}}
                    reqs.append( db.UpdateOne({'_id': keyhash}, cmd, upsert=True) )
                else:
                    Dlost += tout.value
            else:
                Dopret += tout.value

            utxo_pool.update( { utxohash: (tp, tout.value, keyhash) } )

        # work through transaction inputs next / 接下来处理事务输入
        tins = qin_child.get()

        for (tin, utxohash) in tins:
            # pull input out of UTXO pool / 从UTXO池中拉出输入
            (tp, val, keyhash) = utxo_pool.pop(utxohash)
            Dtotal -= val

            if keyhash:
                # try to extract public key from the unlocking script / 尝试从解锁脚本中提取公钥
                (tp, pubkey) = interpret_unlock_script(tp, tin.script.hex)
                # update database / 更新数据库
                cmd = {'$set': {'type': tp}, '$inc': {'val': -val, 'key-seen': int(pubkey != None)}}
                reqs.append( db.UpdateOne({'_id': keyhash}, cmd) )
            else:
                Dlost -= val

        # send requests to writer / 向编写器发送请求
        qchild_write.put(reqs)
        # tell our parent that we are done / 告诉我们的父母我们结束了
        qchild_sync.put( (Dtotal, Dlost, Dopret, len(utxo_pool)) )


    # SYNC PROCESS / 同步过程
# this process synchronizes after the analysis is done / 此过程在分析完成后进行同步
# maintains some status information and initiates the database flushes / 维护一些状态信息并启动数据库刷新
def sync_process(qchild_sync_l, qread_sync, qsync_write, h, utxolen):
    # total number of Satoshis in ciculation / 流通中的中本币总数
    total = 0
    # number of Satoshis sent to OP_RETURN / 发送到OP_RETURN的中本币数量
    opret = 0
    # number of Satoshis at unaccounted addresses / 地址不明的中本聪的数量
    lost = 0

    while True:

        curr_date = qread_sync.get()
        if not curr_date:
            # we are done here / 我们做完了
            # qsync_write.put(False) / 注释掉的代码
            qsync_write.put(True)
            qsync_write.put( (curr_date, total, lost, opret, h.value) )
            # 再放一个False表示真正结束
            qsync_write.put(False)
            return

        utxotot = 0
        # synchronize with children / 与孩子同步
        for qchild in qchild_sync_l:
            (Dtotal, Dlost, Dopret, utxo) = qchild.get()
            # update counters / 更新计数器
            total += Dtotal
            lost += Dlost
            opret += Dopret
            utxotot += utxo

            # update utxo length (for status update only) / 更新utxo长度（仅用于状态更新）
        utxolen.value = utxotot
        h.value += 1

        # send status update to writer process / 发送状态更新到写入进程
        qsync_write.put(True)
        qsync_write.put( (curr_date, total, lost, opret, h.value) )


# WRITER PROCESS / 作家的过程
# this process sends bulk write requests to the database (and waits for their completion) / 此进程向数据库发送批量写请求（并等待它们完成）。
def writer_process(qchild_write_l, qsync_write):
    # flush database after this many blocks (should be smaller than queue length) ... / 在这么多块之后刷新数据库（应该小于队列长度）…
    FLUSH_DATABASE = 1
    # ... but only if number of requests exceeds / …但只有当请求数量超过
    MIN_FLUSH = 1000
    # print status after this many blocks / 在这么多块之后打印状态
    PRINT_STATUS = 150
    # number of blocks after which a snapshot is taken (must be a multiple of PRINT_STATUS and FLUSH_DATABASE) / 快照之后的块数（必须是PRINT_STATUS和FLUSH_DATABASE的倍数）
    TAKE_SNAPSHOT = 3000

    # start mongod client / 启动mongod client
    client = db.MongoClient()
    lookup = client['btc_2']['addr']
    # basic lookup table for bitcoin addresses / 比特币地址的基本查找表
    #   "_id":      public key hash (address can be calculated from this) / 公钥散列（地址可以由此计算）
    #   "type":     address type (P2PK, P2PKH, P2SH) - for initial transaction / 地址类型(P2PK, P2PKH, P2SH) -用于初始交易
    #   "key-seen": number of times public key has been seen / 公钥被看到的次数
    #   "val":      Satoshis at address / 中本聪在地址
    #   "ins":      number of incoming transactions / 传入事务的数目
    #   "outs":     number of outgoing transactions / 传出事务数
    #   "height":   block height where address was last used / 地址最后被使用的块高度
    snapshot = client['btc_2']['snap']
    # snapshot of system state / 系统状态快照

    def take_snapshot(date, total, lost, opret, height):
        snapshot.insert_one({'_id': height, 'date': date, 'tot-val': total, 'op-return': opret, 'unknown': lost })

        # use mongodb aggregate to calculate the numer of active (i.e. with positive balance) addresses and sum of balances for each type / 使用mongodb aggregate来计算每种类型的活跃（即具有正余额）地址的数量和余额之和
        tpsummary = lookup.aggregate([
            {'$match': {'val': {'$gt': 0}}},
            {'$group': {'_id': '$type', 'num-pos': {'$sum': 1}, 'tot-val': {'$sum': '$val'}}}
        ])
        for doc in tpsummary:
            # remove the '_id' item as this is not needed / 删除‘_id’项，因为不需要
            doc['type'] = type2string(doc.pop('_id'))
            snapshot.update_one({'_id': height}, {'$push': {'summary-by-type': doc}})

        # now calculate the number and balance for active addresses where public key is known / 现在计算已知公钥的活动地址的数量和余额
        qsummary = lookup.aggregate([
            {'$match': {'key-seen': {'$gt': 0}, 'val': {'$gt': 0}}},
            {'$group': {'_id': '$type', 'num-pos': {'$sum': 1}, 'tot-val': {'$sum': '$val'}}}
        ])
        qval = 0
        for doc in qsummary:
            # remove the '_id' item as this is not needed / 删除‘_id’项，因为不需要
            doc['type'] = type2string(doc.pop('_id'))
            qval += doc['tot-val']
            snapshot.update_one({'_id': height}, {'$push': {'quantum-by-type': doc}})
        # update the total quantum vulnerable Satoshis / 更新总量子脆弱的中本聪
        snapshot.update_one({'_id': height}, {'$set': {'qattack-frac': float(qval) / float(total), 'unknown-frac': float(lost) / float(total) }})

    reqs = []
    while True:
        # look out for termination signal / 注意终端信号
        if not qsync_write.get():
            # we are done / 我们做完了
            if reqs:
                lookup.bulk_write(reqs, ordered=True, bypass_document_validation=True)
            take_snapshot(date, total, lost, opret, height)
            print('final database update (snapshot taken)')  # 最后的数据库更新（快照）
            return
        # get status / 获得地位
        (date, total, lost, opret, height) = qsync_write.get()

        # receive update requests from both analyzer processes / 接收来自两个分析进程的更新请求
        # transaction outputs go first (so as to avoid negative balances) / 事务输出优先（以避免负余额）
        # and then all the transaction inputs / 然后是所有的交易输入
        for qchild in iter(qchild_write_l):
            reqs.extend(qchild.get())

        if ((height % TAKE_SNAPSHOT == 0) and reqs) or ((height % FLUSH_DATABASE == 0) and (len(reqs) > MIN_FLUSH)):
            # bulk write to database (will halt process until completed) / 批量写入数据库（将暂停进程，直到完成）
            lookup.bulk_write(reqs, ordered=True, bypass_document_validation=True)
            reqs = []

        if height % PRINT_STATUS == 0:
            status = 'database update at height: ' + str(height) + '   block date: ' + str(date)[0:10] + '   cirulating: %.3f kBTC' % (float(total)*1e-11) + '   unknown: %02.2f%%' % (float(lost)/float(total)*100.) + '   op_ret: %.3f mBTC' % (float(opret)*1e-5)
            # take a snapshot of the database approximately every 30 days / 大约每30天对数据库进行一次快照
            if height % TAKE_SNAPSHOT == 0:
                take_snapshot(date, total, lost, opret, height)
                status += '  (snapshot taken)'
            print(status + '\n', end='', flush=True)


# MAIN PROCESS / 主要过程
# queues used to communicate between sub-processes / 用于子进程之间通信的队列

# the reader talks to the output and input analyser, sending them the same block / 阅读器与输出和输入分析器对话，向它们发送相同的块
# we want these queues to be a bit longer so to avoid delays when a new .dat file is opened / 我们希望这些队列稍微长一点，以避免在打开新的.dat文件时出现延迟
qread_unpack_l = [ Queue(12) for i in range(NUM_UNPACKERS) ]
qread_sync = Queue(36)
# pre-processing steps / 预处理步骤
qunpack_hash_l = [ Queue(6) for i in range(NUM_UNPACKERS) ]
qunpack_in_l = [ Queue(6) for i in range(NUM_UNPACKERS) ]
qhash_out = Queue(6)
# the output analyser updates the input analyser and its children when he is done / 输出分析器完成后更新输入分析器及其子分析器
qout_child_l = [ Queue(6*2) for i in range(NUM_ANALYZERS) ]
# input analyser talks to his children / 输入分析器和他的孩子说话
qin_child_l = [ Queue(6) for i in range(NUM_ANALYZERS) ]
qchild_sync_l = [ Queue(6) for i in range(NUM_ANALYZERS) ]
# the children write directly to the writer / 孩子们直接给作者写信
# we want these queues to be a bit longer so that we can process during database snapshots / 我们希望这些队列稍微长一点，以便我们可以在数据库快照期间进行处理
qchild_write_l = [ Queue(36) for i in range(NUM_ANALYZERS) ]
# the synchronizer initiates the flush / 同步器启动刷新
qsync_write = Queue(36 * 2)

# current file, block, and height - shared for status update / 当前文件、块和高度-共享用于状态更新
f = Value('i', 0)
b = Value('i', 0)
h = Value('i', 0)
# utxo length - shared for status update / Utxo长度-共享状态更新
utxolen = Value('i', 0)

# start the four helper processes / 启动四个辅助进程
p_read = Process(target=reader_process, args=(qread_unpack_l, qread_sync, f, b, MAX_FILE))
p_unpack_l = [ Process(target=unpacker_process, args=(qread_unpack_l[i], qunpack_in_l[i], qunpack_hash_l[i])) for i in range(NUM_UNPACKERS) ]
p_hash = Process(target=hasher_process, args=(qunpack_hash_l, qhash_out))
p_dist_in = Process(target=dist_in_process, args=(qunpack_in_l, qin_child_l))
p_dist_out = Process(target=dist_out_process, args=(qhash_out, qout_child_l))
p_analyse_child_l = [ Process(target=analysis_process_child, args=(qin_child_l[i], qout_child_l[i], qchild_sync_l[i], qchild_write_l[i]))
                      for i in range(NUM_ANALYZERS) ]
p_sync = Process(target=sync_process, args=(qchild_sync_l, qread_sync, qsync_write, h, utxolen))
p_write = Process(target=writer_process, args=(qchild_write_l, qsync_write))

T = datetime.now()

if __name__ == "__main__":
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
        # status update on screen (every second) / 屏幕上的状态更新（每秒）
        time.sleep(1)

        delta = datetime.now() - T
        print('running time: %02d:%02d:%02d' % (delta.seconds / (60*60) + delta.days * 24, (delta.seconds / 60) % 60, delta.seconds % 60) + '   input file: ' + str(f.value) + '/' + str(MAX_FILE) + '   height: ' + str(h.value)
              + '/' + str(b.value) + '   utxo size: ' + str(utxolen.value) + '\n', end='', flush=True)
