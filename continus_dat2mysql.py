import base58
import hashlib
import itertools as itr
import pymysql as db
from multiprocessing import Process, Value, Queue
from blockchain_parser.blockchain import Block, get_blocks
import time
from datetime import datetime
from struct import unpack
import os

# /dat/blk00000.dat

# maximum file number (ignore last file because of possible orphans) / 最大文件数（忽略最后一个文件，因为可能有孤儿）
MAX_FILE = 1234
file_path = 'E:/Prof Marco FYP Blockchain/data/blocks/blk%05d.dat'
NUM_UNPACKERS = 3  # 4 cores
NUM_ANALYZERS = 8  # 12 cores (plus reader, hasher, etc.)
QUEUE_SIZE_XLARGE = 100  # Size for large queues


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

# MySQL connection parameters
MYSQL_HOST = 'localhost'
MYSQL_PORT = 3306
MYSQL_USER = 'root'
MYSQL_PASSWORD = ''
MYSQL_DB = 'btc'

def check_and_initialize_db():
    """
    Check if database tables exist and initialize them if needed.
    Returns information about the database state including:
    - tables_exist: whether the tables already exist
    - last_block_hash: hash of the last processed block (if any)
    - last_block_height: height of the last processed block (if any)
    - last_file_number: last processed .dat file number
    - total: total satoshis in circulation
    - lost: satoshis at unaccounted addresses
    - opret: satoshis sent to OP_RETURN
    """
    connection = None
    cursor = None
    try:
        connection = db.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DB,
            charset='utf8mb4'
        )
        cursor = connection.cursor()
        
        # Check if tables exist
        cursor.execute("SHOW TABLES LIKE 'addr'")
        tables_exist = cursor.fetchone() is not None
        
        # Initialize empty results
        result = {
            'tables_exist': tables_exist,
            'last_block_hash': '0000000000000000000000000000000000000000000000000000000000000000',  # Genesis block's previous hash
            'last_block_height': 0,
            'last_file_number': 0,
            'total': 0,
            'lost': 0,
            'opret': 0
        }
        
        if tables_exist:
            # Get last snapshot data
            cursor.execute("SELECT MAX(height) FROM snap")
            max_height_row = cursor.fetchone()
            
            if max_height_row and max_height_row[0]:
                last_height = max_height_row[0]
                
                # Get the snapshot data
                cursor.execute(
                    "SELECT height, date, tot_val, unknown, op_return FROM snap WHERE height = %s",
                    (last_height,)
                )
                snap_data = cursor.fetchone()
                
                if snap_data:
                    result['last_block_height'] = snap_data[0]
                    result['total'] = snap_data[1]
                    result['lost'] = snap_data[2]
                    result['opret'] = snap_data[3]
                    
                    # We also need the last processed block hash for incremental updates
                    # This is stored in a new table we'll create
                    cursor.execute("SHOW TABLES LIKE 'blockchain_state'")
                    if cursor.fetchone() is None:
                        # Create the table if it doesn't exist
                        cursor.execute('''
                        CREATE TABLE blockchain_state (
                            id INT PRIMARY KEY AUTO_INCREMENT,
                            last_processed_height INT,
                            last_processed_hash VARCHAR(64),
                            last_file_number INT,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                        )
                        ''')
                        # Insert a default record with the genesis block's hash as the last_block_hash
                        cursor.execute(
                            "INSERT INTO blockchain_state (last_processed_height, last_processed_hash, last_file_number) VALUES (%s, %s, %s)",
                            (0, '0000000000000000000000000000000000000000000000000000000000000000', 0)
                        )
                        connection.commit()
                    
                    # Retrieve the last processed block hash
                    cursor.execute(
                        "SELECT last_processed_hash, last_file_number FROM blockchain_state ORDER BY id DESC LIMIT 1"
                    )
                    state_data = cursor.fetchone()
                    if state_data:
                        result['last_block_hash'] = state_data[0]
                        result['last_file_number'] = state_data[1]
                    else:
                        # Estimate the last processed .dat file based on block height
                        result['last_file_number'] = estimate_dat_file_number(result['last_block_height'])
                        
        else:
            # Create tables if they don't exist
            print("Database tables not found. Creating new tables...")
            
            # Address table - stores all addresses, their types, and balances
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS addr (
                keyhash BIGINT PRIMARY KEY,
                type INT,
                addr VARCHAR(64),
                val BIGINT,
                key_seen INT DEFAULT 0,
                ins INT DEFAULT 0,
                outs INT DEFAULT 0,
                height INT
            )
            ''')
            
            # Snapshot table - stores blockchain state snapshots at certain block heights
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS snap (
                height INT PRIMARY KEY,
                date DATETIME,
                tot_val BIGINT,
                op_return BIGINT,
                unknown BIGINT,
                qattack_frac FLOAT,
                unknown_frac FLOAT
            )
            ''')
            
            # Summary by address type table - aggregated stats by address type
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS summary_by_type (
                height INT,
                type VARCHAR(20),
                num_pos INT,
                tot_val BIGINT,
                FOREIGN KEY (height) REFERENCES snap(height)
            )
            ''')
            
            # Quantum-vulnerable address stats
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS quantum_by_type (
                height INT,
                type VARCHAR(20),
                num_pos INT,
                tot_val BIGINT,
                FOREIGN KEY (height) REFERENCES snap(height)
            )
            ''')
            
            # UTXO state table - stores the current UTXO state for incremental updates
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS utxo_state (
                utxohash BIGINT PRIMARY KEY,
                type INT,
                value BIGINT,
                keyhash BIGINT
            )
            ''')
            
            # Blockchain state table - stores metadata about the blockchain parsing state
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS blockchain_state (
                id INT PRIMARY KEY AUTO_INCREMENT,
                last_processed_height INT,
                last_processed_hash VARCHAR(64),
                last_file_number INT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            )
            ''')
            
            # Insert initial blockchain state
            cursor.execute(
                "INSERT INTO blockchain_state (last_processed_height, last_processed_hash, last_file_number) VALUES (%s, %s, %s)",
                (0, '0000000000000000000000000000000000000000000000000000000000000000', 0)
            )
            
            connection.commit()
            print("Database tables created successfully.")
        
        return result
    
    except db.Error as e:
        print(f"Database error: {e}")
        # Return default values in case of error
        return {
            'tables_exist': False,
            'last_block_hash': '0000000000000000000000000000000000000000000000000000000000000000',
            'last_block_height': 0,
            'last_file_number': 0,
            'total': 0,
            'lost': 0,
            'opret': 0
        }
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

def estimate_dat_file_number(block_height):
    """
    Estimate the .dat file number based on block height.
    This is an approximation since the exact mapping depends on block sizes.
    """
    # On average, each .dat file contains about 2000 blocks
    # This is a very rough estimate and may need to be adjusted
    return max(0, int(block_height / 2000))

def load_utxo_state(analyzer_count):
    """
    Load the UTXO pool state from the database.
    Returns a list of dictionaries containing UTXOs, grouped by analyzer index.
    
    This function uses batch processing to minimize memory usage and provides
    progress updates during loading.
    
    Args:
        analyzer_count (int): Number of analyzer processes to distribute UTXOs among
        
    Returns:
        list: List of dictionaries containing UTXOs, one dict per analyzer
    """
    utxo_pools = [dict() for _ in range(analyzer_count)]
    connection = None
    cursor = None
    
    try:
        connection = db.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DB,
            charset='utf8mb4'
        )
        cursor = connection.cursor()
        
        # Check if utxo_state table exists
        cursor.execute("SHOW TABLES LIKE 'utxo_state'")
        table_exists = cursor.fetchone() is not None
        
        if not table_exists:
            print("UTXO state table does not exist. Starting with empty UTXO pools.")
            return utxo_pools
            
        # Get total count for progress reporting
        cursor.execute("SELECT COUNT(*) FROM utxo_state")
        total_utxos = cursor.fetchone()[0]
        
        if total_utxos == 0:
            print("UTXO state table is empty. Starting with empty UTXO pools.")
            return utxo_pools
            
        print(f"Loading {total_utxos} UTXOs from database...")
        
        # Load UTXOs in batches to avoid memory issues
        batch_size = 10000
        offset = 0
        loaded_count = 0
        last_progress = 0
        
        while True:
            cursor.execute(
                "SELECT utxohash, type, value, keyhash FROM utxo_state LIMIT %s OFFSET %s",
                (batch_size, offset)
            )
            batch = cursor.fetchall()
            
            if not batch:
                break
                
            for utxohash, tp, value, keyhash in batch:
                # Distribute UTXOs to the appropriate analyzer based on hash
                analyzer_idx = utxohash % analyzer_count
                utxo_pools[analyzer_idx][utxohash] = (tp, value, keyhash)
                loaded_count += 1
                
            # Report progress every 10%
            progress = (loaded_count * 100) // total_utxos
            if progress >= last_progress + 10:
                print(f"Loaded {loaded_count}/{total_utxos} UTXOs ({progress}%)")
                last_progress = progress
                
            offset += batch_size
        
        # Verify all UTXOs were loaded
        actual_loaded = sum(len(pool) for pool in utxo_pools)
        if actual_loaded != total_utxos:
            print(f"Warning: Expected to load {total_utxos} UTXOs but loaded {actual_loaded}")
        
        # Distribution statistics
        min_size = min(len(pool) for pool in utxo_pools)
        max_size = max(len(pool) for pool in utxo_pools)
        avg_size = actual_loaded / len(utxo_pools)
        print(f"UTXO distribution - Min: {min_size}, Avg: {avg_size:.1f}, Max: {max_size}")
        
        return utxo_pools
        
    except db.Error as e:
        print(f"Database error loading UTXO state: {e}")
        print("Starting with empty UTXO pools.")
        return [dict() for _ in range(analyzer_count)]
        
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

def update_blockchain_state(height, block_hash, file_number):
    """
    Update the blockchain_state table with the latest processed block information.
    This is crucial for incremental updates to start from the correct block.
    
    Args:
        height (int): The block height
        block_hash (str): The hash of the block
        file_number (int): The .dat file number containing this block
    """
    try:
        connection = db.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DB,
            charset='utf8mb4'
        )
        cursor = connection.cursor()
        
        # Update the blockchain state
        cursor.execute(
            "UPDATE blockchain_state SET last_processed_height = %s, last_processed_hash = %s, last_file_number = %s",
            (height, block_hash, file_number)
        )
        connection.commit()
        
        cursor.close()
        connection.close()
    except db.Error as e:
        print(f"Error updating blockchain state: {e}")

def save_utxo_state(utxo_pools):
    """
    Save the current UTXO pool state to the database.
    Uses efficient batching to minimize memory usage.
    
    Args:
        utxo_pools (list): List of dictionaries containing UTXOs
    """
    try:
        connection = db.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DB,
            charset='utf8mb4'
        )
        cursor = connection.cursor()
        
        # Clear the current UTXO state - use a transaction for safety
        cursor.execute("START TRANSACTION")
        cursor.execute("TRUNCATE TABLE utxo_state")
        
        # Save the current UTXOs in batches to minimize memory usage
        batch = []
        batch_size = 5000  # Increased batch size for better performance
        total_utxos = 0
        
        print("Starting UTXO state save...")
        
        for pool_idx, pool in enumerate(utxo_pools):
            pool_size = len(pool)
            total_utxos += pool_size
            print(f"Processing UTXO pool {pool_idx+1}/{len(utxo_pools)} with {pool_size} UTXOs")
            
            for utxohash, (tp, value, keyhash) in pool.items():
                batch.append((utxohash, tp, value, keyhash))
                
                if len(batch) >= batch_size:
                    cursor.executemany(
                        "INSERT INTO utxo_state (utxohash, type, value, keyhash) VALUES (%s, %s, %s, %s)",
                        batch
                    )
                    batch = []
                    # Commit every few batches to avoid transaction getting too large
                    if len(batch) % (batch_size * 10) == 0:
                        connection.commit()
                        cursor.execute("START TRANSACTION")
        
        # Insert any remaining UTXOs
        if batch:
            cursor.executemany(
                "INSERT INTO utxo_state (utxohash, type, value, keyhash) VALUES (%s, %s, %s, %s)",
                batch
            )
        
        # Commit the transaction
        connection.commit()
        print(f"Saved {total_utxos} UTXOs to database")
        
        cursor.close()
        connection.close()
        
    except db.Error as e:
        print(f"Error saving UTXO state: {e}")
        # If there's an error, try to rollback the transaction
        try:
            if connection and connection.open:
                connection.rollback()
        except:
            pass
        raise

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
        try:
            for i in range(0, lock[-2]-80):
                # Check if a valid key / 检查是否有有效的密钥
                l = hexkeys[0]
                if l in [33, 65] and len(hexkeys) > l:
                    keys.append(hexkeys[1:l+1].hex())
                    hexkeys = hexkeys[l+1:]
                else:
                    return (ADDR_UNKNOWN, None, None, None)
            if len(hexkeys) == 0:
                return (ADDR_MULTISIG, calcshorthash(keyhash), addr_from_keyhash(keyhash, b'\x05'), keys)
            else:
                return (ADDR_UNKNOWN, None, None, None)
        except (IndexError, ValueError) as e:
            # Handle potential errors in the multisig parsing
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
def reader_process(qread_unpack_l, qread_sync, f, b, fmax, start_hash='0000000000000000000000000000000000000000000000000000000000000000'):
    """
    Process that reads blocks from .dat files and feeds them to analysis processes.
    
    This function:
    1. Starts from the specified block hash (or genesis block if not specified)
    2. Maintains a pool of blocks to ensure they're processed in the correct order
    3. Reads new .dat files when needed
    4. Sends blocks and their timestamps to downstream processes
    
    Args:
        qread_unpack_l: List of queues for sending blocks to unpacker processes
        qread_sync: Queue for sending block timestamps to sync process
        f: Shared Value for current file number
        b: Shared Value for block counter
        fmax: Maximum file number to process
        start_hash: Hash of the last processed block (to start from)
    """
    # previous hash is set to the last processed block or genesis block / 前一个哈希设置为最后处理的块或创世块
    curr_hash = start_hash
    # block pool / 块池
    # indexed by previous hash / 按前一个散列索引
    blk_pool = {}
    # open first file / 打开第一个文件
    rdr = iter([])
    h = 0
    
    print(f"Starting block processing from block with previous hash: {curr_hash}")
    print(f"Starting with .dat file number: {f.value}")

    while True:
        # if next block is not currently in the pool, read the next file / 如果下一个块当前不在池中，则读取下一个文件
        if curr_hash in blk_pool:
            blk = blk_pool.pop(curr_hash)
            curr_hash = blk.hash
            
            # Store current block hash for snapshot updates
            qhash = (curr_hash, f.value)
            
            # unpack block in next process / 在下一个进程中解包块
            qread_unpack_l[h % 3].put(blk)
            h += 1
            
            # submit block info and hash for database pushes / 为数据库推送提交块信息和哈希
            qread_sync.put((blk.header.timestamp, qhash))
            
            # Print progress updates periodically
            if h % 1000 == 0:
                print(f"Processed {h} blocks, current hash: {curr_hash}")
        else:
            try:
                blk = Block(next(rdr))
                # update pool / 更新池
                blk_pool.update({blk.header.previous_block_hash: blk})
                # increment block counter / 增量块计数器
                b.value += 1
            except StopIteration:
                # time to move to the next file / 是时候转到下一个文件了
                if f.value < fmax:
                    try:
                        dat_file_path = file_path % f.value
                        print(f"Opening new .dat file: {dat_file_path}")
                        rdr = iter(get_blocks(dat_file_path))
                        f.value += 1
                    except Exception as e:
                        print(f"Error opening .dat file {f.value}: {e}")
                        if f.value < fmax - 1:
                            # Try the next file
                            f.value += 1
                            continue
                        else:
                            # We can't read any more files
                            print("Cannot read any more .dat files, terminating")
                            # Signal end of processing
                            for q in qread_unpack_l:
                                q.put(None)
                            qread_sync.put(None)
                            return
                else:
                    # we are done here / 我们做完了
                    print(f"Reached maximum file number {fmax}, terminating")
                    # let the analysis process know / 让分析过程知道
                    for q in qread_unpack_l:
                        q.put(None)
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
def analysis_process_child(qin_child, qout_child, qchild_sync, qchild_write, initial_utxo_pool=None):
    # unspent transaction outputs / 未使用的事务输出
    # indexed by both transaction hash and output index / 由事务哈希和输出索引索引
    utxo_pool = initial_utxo_pool or {}

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
                    # MySQL equivalent of MongoDB's UpdateOne with upsert
                    # Create a structure that represents the SQL operation to be performed
                    cmd = {
                        '_id': keyhash,
                        '$setOnInsert': {'type': tp, 'addr': addr},
                        '$inc': {'val': tout.value, 'key-seen': int(pubkey != None)}
                    }
                    reqs.append(cmd)
                else:
                    Dlost += tout.value
            else:
                Dopret += tout.value

            utxo_pool.update( { utxohash: (tp, tout.value, keyhash) } )

        # work through transaction inputs next / 接下来处理事务输入
        tins = qin_child.get()

        for (tin, utxohash) in tins:
            # pull input out of UTXO pool / 从UTXO池中拉出输入
            if utxohash in utxo_pool:
                (tp, val, keyhash) = utxo_pool.pop(utxohash)
                Dtotal -= val

                if keyhash:
                    # try to extract public key from the unlocking script / 尝试从解锁脚本中提取公钥
                    (tp, pubkey) = interpret_unlock_script(tp, tin.script.hex)
                    # MySQL equivalent of MongoDB's UpdateOne
                    cmd = {
                        '_id': keyhash,
                        '$set': {'type': tp},
                        '$inc': {'val': -val, 'key-seen': int(pubkey != None)}
                    }
                    reqs.append(cmd)
                else:
                    Dlost -= val
            else:
                # This can happen if we're doing incremental updates
                # and don't have the complete UTXO pool state
                print(f"Warning: UTXO {utxohash} not found in pool")

        # send requests to writer / 向编写器发送请求
        qchild_write.put(reqs)
        # tell our parent that we are done / 告诉我们的父母我们结束了
        qchild_sync.put( (Dtotal, Dlost, Dopret, len(utxo_pool)) )


# SYNC PROCESS / 同步过程
# this process synchronizes after the analysis is done / 此过程在分析完成后进行同步
# maintains some status information and initiates the database flushes / 维护一些状态信息并启动数据库刷新
def sync_process(qchild_sync_l, qread_sync, qsync_write, h, utxolen, initial_total=0, initial_opret=0, initial_lost=0):
    """
    Synchronizes after block analysis is complete and maintains state.
    
    This process:
    1. Tracks global counters (total satoshis, lost, OP_RETURN, etc.)
    2. Collects data from child analyzer processes
    3. Updates shared status variables
    4. Sends status updates to the writer process
    
    Args:
        qchild_sync_l: List of queues for receiving data from analyzer children
        qread_sync: Queue for receiving block timestamps from reader
        qsync_write: Queue for sending data to writer
        h: Shared Value for block height counter
        utxolen: Shared Value for UTXO pool size
        initial_total: Initial value for total satoshis in circulation
        initial_opret: Initial value for satoshis sent to OP_RETURN
        initial_lost: Initial value for satoshis at unaccounted addresses
    """
    # total number of Satoshis in circulation / 流通中的中本币总数
    total = initial_total
    # number of Satoshis sent to OP_RETURN / 发送到OP_RETURN的中本币数量
    opret = initial_opret
    # number of Satoshis at unaccounted addresses / 地址不明的中本聪的数量
    lost = initial_lost
    
    print(f"Sync process starting with: Total: {total}, Lost: {lost}, OP_RETURN: {opret}")
 
    while True:
        # Get data from reader process
        data = qread_sync.get()
        if data is None:
            # we are done here / 我们做完了
            print("Received termination signal in sync process")
            qsync_write.put(True)
            qsync_write.put((None, total, lost, opret, h.value, None))  # No date available
            # 再放一个False表示真正结束
            qsync_write.put(False)
            return
            
        # Unpack the data - now includes block hash
        curr_date, block_hash_info = data

        utxotot = 0
        # synchronize with children / 与孩子同步
        for qchild in qchild_sync_l:
            data = qchild.get()
            if data:
                Dtotal, Dlost, Dopret, utxo = data
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
        qsync_write.put((curr_date, total, lost, opret, h.value, block_hash_info))


# WRITER PROCESS / 作家的过程
# this process sends bulk write requests to the database (and waits for their completion) / 此进程向数据库发送批量写请求（并等待它们完成）。
def writer_process(qchild_write_l, qsync_write, utxo_pools=None, f=None):
    """
    Process that handles database writes, including address updates and snapshots.
    
    This process:
    1. Receives update requests from analyzer processes
    2. Performs database operations in batches
    3. Takes periodic snapshots of the blockchain state
    4. Saves UTXO state during snapshots
    5. Updates blockchain state tracking
    
    Args:
        qchild_write_l: List of queues for receiving requests from analyzer processes
        qsync_write: Queue for receiving status updates from sync process
        utxo_pools: List of UTXO pools for saving state
        f: Shared Value for current file number (for blockchain state updates)
    """
    # flush database after this many blocks (should be smaller than queue length) ... / 在这么多块之后刷新数据库（应该小于队列长度）…
    FLUSH_DATABASE = 1
    # ... but only if number of requests exceeds / …但只有当请求数量超过
    MIN_FLUSH = 1000
    # print status after this many blocks / 在这么多块之后打印状态
    PRINT_STATUS = 150
    # number of blocks after which a snapshot is taken (must be a multiple of PRINT_STATUS and FLUSH_DATABASE) / 快照之后的块数（必须是PRINT_STATUS和FLUSH_DATABASE的倍数）
    TAKE_SNAPSHOT = 3000
    
    # Connect to MySQL
    connection = db.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DB,
        charset='utf8mb4'
    )
    cursor = connection.cursor()

    def take_snapshot(date, total, lost, opret, height, block_hash_info=None):        
        """
        Takes a snapshot of the blockchain state at the given height.
        
        Args:
            date: Timestamp of the current block
            total: Total satoshis in circulation
            lost: Satoshis at unaccounted addresses
            opret: Satoshis sent to OP_RETURN
            height: Current block height
            block_hash_info: Tuple of (block_hash, file_number) if available
        """
        try:
            # Insert snapshot data
            cursor.execute(
                "INSERT INTO snap (height, date, tot_val, op_return, unknown, qattack_frac, unknown_frac) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (height, date, total, opret, lost, 0.0, float(lost) / float(total) if total > 0 else 0.0)
            )
            connection.commit()

            # Calculate summary by type
            cursor.execute('''
            SELECT type, COUNT(*) as num_pos, SUM(val) as tot_val 
            FROM addr 
            WHERE val > 0 
            GROUP BY type
            ''')
            
            for row in cursor.fetchall():
                type_int, num_pos, tot_val = row
                type_str = type2string(type_int)
                cursor.execute(
                    "INSERT INTO summary_by_type (height, type, num_pos, tot_val) VALUES (%s, %s, %s, %s)",
                    (height, type_str, num_pos, tot_val)
                )
            
            # Calculate quantum vulnerable addresses
            cursor.execute('''
            SELECT type, COUNT(*) as num_pos, SUM(val) as tot_val 
            FROM addr 
            WHERE key_seen > 0 AND val > 0 
            GROUP BY type
            ''')
            
            qval = 0
            for row in cursor.fetchall():
                type_int, num_pos, tot_val = row
                type_str = type2string(type_int)
                qval += tot_val
                cursor.execute(
                    "INSERT INTO quantum_by_type (height, type, num_pos, tot_val) VALUES (%s, %s, %s, %s)",
                    (height, type_str, num_pos, tot_val)
                )
            
            # Update quantum attack vulnerability
            cursor.execute(
                "UPDATE snap SET qattack_frac = %s WHERE height = %s",
                (float(qval) / float(total) if total > 0 else 0.0, height)
            )
            connection.commit()
            
            # Save UTXO state if provided
            if utxo_pools:
                save_utxo_state(utxo_pools)
                
            # Update blockchain state with current block hash if available
            if block_hash_info:
                block_hash, file_number = block_hash_info
                update_blockchain_state(height, block_hash, file_number)
                
            print(f"Snapshot taken at height {height}" + 
                  (f", block hash: {block_hash_info[0][:8]}..." if block_hash_info else ""))
            
        except db.Error as e:
            print(f"Error taking snapshot: {e}")
            # Don't re-raise to allow processing to continue

    # SQL statements for database operations
    insert_addr_sql = '''
    INSERT INTO addr (keyhash, type, addr, val, key_seen)
    VALUES (%s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
    val = val + VALUES(val),
    key_seen = key_seen + VALUES(key_seen)
    '''
    
    update_addr_sql = '''
    UPDATE addr
    SET type = %s, val = val + %s, key_seen = key_seen + %s
    WHERE keyhash = %s
    '''

    reqs = []
    last_block_hash_info = None
    
    try:
        while True:
            # look out for termination signal / 注意终端信号
            if not qsync_write.get():
                # we are done / 我们做完了
                print("Writer process received termination signal")
                if reqs: 
                    # Execute any remaining requests
                    print(f"Executing final {len(reqs)} pending requests")
                    try:
                        for req in reqs:
                            if req.get('$setOnInsert', None):
                                # This is an insert or update operation
                                keyhash = req['_id']
                                data = req['$setOnInsert']
                                inc_data = req.get('$inc', {})
                                
                                cursor.execute(insert_addr_sql, 
                                    (keyhash, 
                                     data.get('type', None), 
                                     data.get('addr', None), 
                                     inc_data.get('val', 0),
                                     inc_data.get('key-seen', 0))
                                )
                            elif req.get('$set', None) or req.get('$inc', None):
                                # This is an update operation
                                keyhash = req['_id']
                                set_data = req.get('$set', {})
                                inc_data = req.get('$inc', {})
                                
                                cursor.execute(update_addr_sql,
                                    (set_data.get('type', None),
                                     inc_data.get('val', 0),
                                     inc_data.get('key-seen', 0),
                                     keyhash)
                                )
                        connection.commit()
                    except db.Error as e:
                        print(f"Error executing final requests: {e}")
                
                # Take final snapshot
                if date and height:
                    take_snapshot(date, total, lost, opret, height, last_block_hash_info)
                    print('Final database update complete (snapshot taken)')
                
                break
                
            # get status / 获得地位
            date, total, lost, opret, height, block_hash_info = qsync_write.get()
            
            # Store the block hash info for final snapshot
            if block_hash_info:
                last_block_hash_info = block_hash_info

            # receive update requests from analyzer processes
            for qchild in iter(qchild_write_l):
                child_reqs = qchild.get()
                if child_reqs:
                    reqs.extend(child_reqs)

            # Execute batch requests if enough have accumulated or it's time for a snapshot
            if ((height % TAKE_SNAPSHOT == 0) and reqs) or ((height % FLUSH_DATABASE == 0) and (len(reqs) > MIN_FLUSH)):
                # Execute batch of requests
                batch_start_time = time.time()
                try:
                    for req in reqs:
                        if req.get('$setOnInsert', None):
                            # This is an insert or update operation
                            keyhash = req['_id']
                            data = req['$setOnInsert']
                            inc_data = req.get('$inc', {})
                            
                            cursor.execute(insert_addr_sql, 
                                (keyhash, 
                                 data.get('type', None), 
                                 data.get('addr', None), 
                                 inc_data.get('val', 0),
                                 inc_data.get('key-seen', 0))
                            )
                        elif req.get('$set', None) or req.get('$inc', None):
                            # This is an update operation
                            keyhash = req['_id']
                            set_data = req.get('$set', {})
                            inc_data = req.get('$inc', {})
                            
                            cursor.execute(update_addr_sql,
                                (set_data.get('type', None),
                                 inc_data.get('val', 0),
                                 inc_data.get('key-seen', 0),
                                 keyhash)
                            )
                    connection.commit()
                    batch_time = time.time() - batch_start_time
                    print(f"Executed batch of {len(reqs)} requests in {batch_time:.2f}s")
                    reqs = []
                except db.Error as e:
                    print(f"Error executing batch requests: {e}")
                    # Don't clear reqs here to retry later

            if height % PRINT_STATUS == 0:
                status = 'database update at height: ' + str(height) + '   block date: ' + str(date)[0:10] + '   cirulating: %.3f kBTC' % (float(total)*1e-11) + '   unknown: %02.2f%%' % (float(lost)/float(total)*100.) + '   op_ret: %.3f mBTC' % (float(opret)*1e-5)
                # take a snapshot of the database approximately every 30 days / 大约每30天对数据库进行一次快照
                if height % TAKE_SNAPSHOT == 0:
                    take_snapshot(date, total, lost, opret, height, block_hash_info)
                    status += '  (snapshot taken)'
                print(status + '\n', end='', flush=True)
    finally:
        if cursor:
            cursor.close()
        if connection and connection.open:
            connection.close()
        print("Writer process finished")


# MAIN PROCESS / 主要过程
# Modified to support incremental updates
if __name__ == "__main__":
    try:
        print("Bitcoin blockchain incremental analysis tool")
        print("============================================")
        
        # Initialize database and get previous state
        print("Checking database state...")
        db_state = check_and_initialize_db()
        
        # Load UTXO pool state if tables exist
        utxo_pools = None
        if db_state['tables_exist']:
            print(f"Found existing database.")
            print(f"Last processed block height: {db_state['last_block_height']}")
            print(f"Last processed block hash: {db_state['last_block_hash']}")
            print(f"Estimated last processed .dat file: {db_state['last_file_number']}")
            
            if db_state['last_block_height'] > 0:
                print(f"Loaded circulation stats - Total: {db_state['total']}, Lost: {db_state['lost']}, OP_RETURN: {db_state['opret']}")
                print(f"Loading UTXO pool state...")
                utxo_pools = load_utxo_state(NUM_ANALYZERS)
                total_utxos = sum(len(pool) for pool in utxo_pools) if utxo_pools else 0
                print(f"Loaded {total_utxos} UTXOs")
            else:
                print("No previous blockchain data found. Starting from genesis block.")
        else:
            print("No existing database found. Tables will be created.")
        
        # Get .dat file information
        max_file = MAX_FILE
        try:
            # Try to dynamically find the maximum file number by checking what files exist
            highest_file = -1
            for i in range(MAX_FILE + 1):
                test_path = file_path % i
                if os.path.exists(test_path):
                    highest_file = i
            
            if highest_file >= 0:
                max_file = highest_file
                print(f"Found .dat files up to {max_file} (blk{max_file:05d}.dat)")
            else:
                print(f"Warning: No .dat files found at path pattern {file_path}")
                print(f"Using configured maximum file: {max_file}")
        except Exception as e:
            print(f"Error detecting .dat files: {e}")
            print(f"Using configured maximum file: {max_file}")
            
        # queues used to communicate between sub-processes / 用于子进程之间通信的队列
        print("Setting up process communication...")
        
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
        qsync_write = Queue(QUEUE_SIZE_XLARGE * 2)
        
        # current file, block, and height - shared for status update / 当前文件、块和高度-共享用于状态更新
        # Initialize with values from database if incremental
        start_file = db_state['last_file_number'] if db_state['tables_exist'] and db_state['last_block_height'] > 0 else 0
        f = Value('i', start_file)
        b = Value('i', 0)  # Will be updated during processing
        h = Value('i', db_state['last_block_height'] if db_state['tables_exist'] and db_state['last_block_height'] > 0 else 0)
        # utxo length - shared for status update / Utxo长度-共享状态更新
        utxolen = Value('i', sum(len(pool) for pool in utxo_pools) if utxo_pools else 0)
        
        print("Starting processing processes...")
        # start the processing processes / 启动处理进程
        p_read = Process(target=reader_process, args=(qread_unpack_l, qread_sync, f, b, max_file, db_state['last_block_hash']))
        p_unpack_l = [ Process(target=unpacker_process, args=(qread_unpack_l[i], qunpack_in_l[i], qunpack_hash_l[i])) for i in range(NUM_UNPACKERS) ]
        p_hash = Process(target=hasher_process, args=(qunpack_hash_l, qhash_out))
        p_dist_in = Process(target=dist_in_process, args=(qunpack_in_l, qin_child_l))
        p_dist_out = Process(target=dist_out_process, args=(qhash_out, qout_child_l))
        
        # Initialize analyzer processes with UTXO pools if incremental
        p_analyse_child_l = []
        for i in range(NUM_ANALYZERS):
            initial_pool = utxo_pools[i] if utxo_pools else None
            p = Process(target=analysis_process_child, args=(qin_child_l[i], qout_child_l[i], qchild_sync_l[i], qchild_write_l[i], initial_pool))
            p_analyse_child_l.append(p)
        
        # Initialize sync process with values from database if incremental
        p_sync = Process(target=sync_process, args=(
            qchild_sync_l, 
            qread_sync, 
            qsync_write, 
            h, 
            utxolen, 
            db_state['total'] if db_state['tables_exist'] and db_state['last_block_height'] > 0 else 0,
            db_state['opret'] if db_state['tables_exist'] and db_state['last_block_height'] > 0 else 0,
            db_state['lost'] if db_state['tables_exist'] and db_state['last_block_height'] > 0 else 0
        ))
        
        # Initialize writer process with UTXO pools reference for saving state
        p_write = Process(target=writer_process, args=(qchild_write_l, qsync_write, utxo_pools, f))
        
        T = datetime.now()
        
        # Start all processes
        print("Starting reader process...")
        p_read.start()
        
        print("Starting unpacker processes...")
        for i, process in enumerate(p_unpack_l):
            process.start()
            
        print("Starting hasher process...")
        p_hash.start()
        
        print("Starting distribution processes...")
        p_dist_in.start()
        p_dist_out.start()
        
        print("Starting analyzer processes...")
        for i, process in enumerate(p_analyse_child_l):
            process.start()
            
        print("Starting sync process...")
        p_sync.start()
        
        print("Starting writer process...")
        p_write.start()
        
        print("\nAll processes started. Processing blockchain...")
        
        # Monitor processes and display progress
        while p_write.is_alive():
            # status update on screen (every second) / 屏幕上的状态更新（每秒）
            time.sleep(1)
        
            delta = datetime.now() - T
            hours = delta.seconds // 3600 + delta.days * 24
            minutes = (delta.seconds // 60) % 60
            seconds = delta.seconds % 60
            
            print(f'Running time: {hours:02d}:{minutes:02d}:{seconds:02d}   File: {f.value}/{max_file}   Height: {h.value}/{b.value}   UTXO size: {utxolen.value}\n', 
                  end='', flush=True)
        
        # Wait for all processes to finish
        print("\nWriter process completed. Waiting for other processes to exit...")
        
        # Wait for processes with timeout
        for process, name in [
            (p_read, "Reader"), 
            (p_hash, "Hasher"), 
            (p_dist_in, "Input Distributor"), 
            (p_dist_out, "Output Distributor"),
            (p_sync, "Sync")
        ]:
            process.join(timeout=5)
            if process.is_alive():
                print(f"{name} process still running, terminating...")
                process.terminate()
                
        for i, process in enumerate(p_unpack_l):
            process.join(timeout=3)
            if process.is_alive():
                print(f"Unpacker {i} still running, terminating...")
                process.terminate()
                
        for i, process in enumerate(p_analyse_child_l):
            process.join(timeout=3)
            if process.is_alive():
                print(f"Analyzer {i} still running, terminating...")
                process.terminate()
        
        print("\nProcessing complete!")
        total_time = datetime.now() - T
        hours = total_time.seconds // 3600 + total_time.days * 24
        minutes = (total_time.seconds // 60) % 60
        seconds = total_time.seconds % 60
        print(f"Total run time: {hours:02d}:{minutes:02d}:{seconds:02d}")
        print(f"Processed blocks: {h.value}")
        print(f"Final UTXO size: {utxolen.value}")
    
    except KeyboardInterrupt:
        print("\nUser interrupted processing. Terminating processes...")
        for process in [p_read, p_hash, p_dist_in, p_dist_out, p_sync, p_write] + p_unpack_l + p_analyse_child_l:
            if process.is_alive():
                process.terminate()
        print("All processes terminated.")
        
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        # Terminate any running processes
        for process in [p_read, p_hash, p_dist_in, p_dist_out, p_sync, p_write] + p_unpack_l + p_analyse_child_l:
            if 'process' in locals() and process.is_alive():
                process.terminate()
