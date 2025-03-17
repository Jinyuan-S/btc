from fastapi import FastAPI, HTTPException
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, func, and_, or_, desc, text
from app.config import settings
from app.models import Block, Transaction, TxInput, TxOutput, Address
from contextlib import asynccontextmanager
from typing import List, Dict, Any
from pydantic import BaseModel, Field
import logging
import uvicorn
from fastapi import Response, HTTPException
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import hashlib
import base58
from sqlalchemy.exc import ProgrammingError, OperationalError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Address type constants
ADDR_UNKNOWN = int(-1)
ADDR_P2PK = int(1)      # pay to public key
ADDR_P2PK_comp = int(2)  # pay to compressed public key
ADDR_MULTISIG = int(3)   # multisignature (any variant)
ADDR_P2PKH = int(10)     # pay to public key hash
ADDR_P2SH = int(20)      # pay to script hash (not yet known - unspent)
ADDR_P2SH_PK = int(21)   # pay to script hash (P2PK script)
ADDR_P2SH_MULTISIG = int(22)  # pay to script hash (multisig script)
ADDR_P2SH_OTHER = int(23)     # pay to script hash (other script)
ADDR_OPRETURN = int(0)        # op_return - unspendable

def type2string(tp):
    """Convert address type code to string representation"""
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
    """Calculate address from key hash"""
    checksum = hashlib.sha256(hashlib.sha256(b'\x00' + keyhash).digest()).digest()
    return base58.b58encode(b'\x00' + keyhash + checksum[:4])

def interpret_lock_script(lock):
    """Analyze locking script to determine address type and extract keyhash"""
    def calcshorthash(keyhash):
        return int.from_bytes(keyhash[:8], byteorder='big', signed=False) >> 1

    def addr_from_keyhash(keyhash, prefix):
        checksum = hashlib.sha256(hashlib.sha256(prefix + keyhash).digest()).digest()
        return base58.b58encode(prefix + keyhash + checksum[:4])

    # Convert lock to bytes if it's hex string
    if isinstance(lock, str):
        lock = bytes.fromhex(lock)
    
    # P2PK format
    if ((len(lock) == 67 and lock.hex()[0:2] == '41') or (len(lock) == 35 and lock.hex()[:2] == '21')) and lock.hex()[-2:] == 'ac':
        key = (lock[1:-1]).hex()
        keyhash = hashlib.new('ripemd160', hashlib.sha256(lock[1:-1]).digest()).digest()
        if len(lock) == 67:
            tp = ADDR_P2PK
        else:
            tp = ADDR_P2PK_comp
        return (tp, calcshorthash(keyhash), addr_from_keyhash(keyhash, b'\x00'), key)

    # P2PKH format
    elif len(lock) == 25 and lock.hex()[:6] == '76a914' and lock.hex()[-4:] == '88ac':
        keyhash = lock[3:-2]
        return (ADDR_P2PKH, calcshorthash(keyhash), addr_from_keyhash(keyhash, b'\x00'), None)

    # P2SH format
    elif len(lock) == 23 and lock.hex()[:4] == 'a914' and lock.hex()[-2:] == '87':
        keyhash = lock[2:-1]
        return (ADDR_P2SH, calcshorthash(keyhash), addr_from_keyhash(keyhash, b'\x05'), None)

    # OP_RETURN
    elif len(lock) > 1 and lock.hex()[:2] == '6a':
        return (ADDR_OPRETURN, None, None, None)

    # MULTISIG
    elif len(lock) >= 36 and lock.hex()[-2:] == 'ae':
        keyhash = hashlib.new('ripemd160', hashlib.sha256(lock).digest()).digest()
        keys = []
        hexkeys = lock[1:-2]
        for i in range(0, lock[-2] - 80):
            try:
                l = hexkeys[0]
                if l in [33, 65] and len(hexkeys) > l:
                    keys.append(hexkeys[1:l+1].hex())
                    hexkeys = hexkeys[l+1:]
                else:
                    return (ADDR_UNKNOWN, None, None, None)
            except Exception:
                return (ADDR_UNKNOWN, None, None, None)
        if len(hexkeys) == 0:
            return (ADDR_MULTISIG, calcshorthash(keyhash), addr_from_keyhash(keyhash, b'\x05'), keys)
        else:
            return (ADDR_UNKNOWN, None, None, None)

    # UNKNOWN
    else:
        return (ADDR_UNKNOWN, None, None, None)

def interpret_unlock_script(tp, unlock):
    """Analyze unlocking script to determine if public key is revealed"""
    # Convert unlock to bytes if it's hex string
    if isinstance(unlock, str):
        unlock = bytes.fromhex(unlock)
        
    if tp == ADDR_P2PKH:
        if len(unlock) > 2:
            return (tp, unlock[2+unlock[0]:].hex())
        return (tp, None)
    elif tp == ADDR_P2SH:
        lock = []  # Take the lock script to be the last data pushed
        try:
            while len(unlock) > 0:
                # Regular opcode
                if unlock[0] == 0 or unlock[0] > 78:
                    unlock = unlock[1:]
                # PUSHDATA1
                elif unlock[0] == 76:
                    l = int.from_bytes(unlock[1:2], byteorder='little')
                    lock = unlock[2:l+2]
                    unlock = unlock[l+2:]
                # PUSHDATA2
                elif unlock[0] == 77:
                    l = int.from_bytes(unlock[1:3], byteorder='little')
                    lock = unlock[3:l+3]
                    unlock = unlock[l+3:]
                # PUSHDATA4, can't actually be used
                elif unlock[0] == 78:
                    l = int.from_bytes(unlock[1:5], byteorder='little')
                    lock = unlock[5:l+5]
                    unlock = unlock[l+5:]
                # Other PUSHDATA
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
        except Exception:
            return (ADDR_P2SH_OTHER, None)
    else:
        return (tp, None)

# Check if tables exist in the database
async def check_tables_exist(session):
    """Check if required tables exist in the database"""
    try:
        # Try to query the tables to see if they exist
        await session.execute(text("SELECT 1 FROM blocks LIMIT 1"))
        await session.execute(text("SELECT 1 FROM transactions LIMIT 1"))
        await session.execute(text("SELECT 1 FROM tx_inputs LIMIT 1"))
        await session.execute(text("SELECT 1 FROM tx_outputs LIMIT 1"))
        return True
    except (ProgrammingError, OperationalError) as e:
        logger.error(f"Database tables not found: {e}")
        return False

# Create connection to the database
try:
    engine = create_async_engine(
        settings.MYSQL_URL,
        echo=True,  # 打印SQL语句
        pool_size=5,    # 连接池大小
        max_overflow=10,    # 溢出连接池大小
        pool_timeout=30,    # 连接池超时时间
        pool_recycle=1800,  # 连接池回收时间
    )
except Exception as e:
    logger.error(f"Failed to create engine: {e}")

try:
    # 创建异步会话工厂
    async_session = sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
except Exception as e:
    logger.error(f"Failed to create sessionmaker: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.engine = engine
    app.async_session = async_session
    yield
    # Shutdown
    await engine.dispose()

app = FastAPI(
    title="Bitcoin Address Analysis API",
    description="""
    比特币地址分析API，提供地址类型分布的历史数据和当前摘要信息。
    
    主要功能：
    * 获取比特币地址类型分布的历史数据
    * 获取当前比特币地址类型的分布摘要
    * 分析量子计算对比特币网络的潜在影响
    
    所有金额数据均以kBTC（千比特币）为单位。
    """,
    version="1.0.0",
    contact={
        "name": "Your Name",
        "email": "your.email@example.com",
    },
    lifespan=lifespan,
)

class AddressHistoryResponse(BaseModel):
    """
    比特币地址历史数据响应模型
    """
    height: List[int] = Field(
        description="区块高度列表，每3000个区块一个采样点",
        example=[3000, 6000, 9000]
    )
    total_value: List[float] = Field(
        description="对应区块高度的比特币总流通量（单位：kBTC）",
        example=[150.0, 300.0, 450.0]
    )
    p2pk_value: List[float] = Field(
        description="P2PK类型地址（包括multisig和compressed）持有的比特币数量（单位：kBTC）",
        example=[146.02308, 288.97308, 429.37275]
    )
    revealed_value: List[float] = Field(
        description="已知公钥的地址持有的比特币数量（量子易受攻击部分）（单位：kBTC）",
        example=[146.08699, 289.03699, 429.43666]
    )
    pot_revealed_value: List[float] = Field(
        description="潜在暴露的比特币数量（包括已知公钥和P2SH unknown）（单位：kBTC）",
        example=[146.08699, 289.03699, 429.43666]
    )

    class Config:
        schema_extra = {
            "example": {
                "height": [3000, 6000, 9000],
                "total_value": [150.0, 300.0, 450.0],
                "p2pk_value": [146.02308, 288.97308, 429.37275],
                "revealed_value": [146.08699, 289.03699, 429.43666],
                "pot_revealed_value": [146.08699, 289.03699, 429.43666]
            }
        }

class AddressSummaryResponse(BaseModel):
    """
    比特币地址分布摘要响应模型
    """
    p2pk_total: float = Field(
        description="P2PK类型地址（包括multisig和compressed）持有的总比特币数量（单位：kBTC）",
        example=1763.15081330363
    )
    quantum_vulnerable_minus_p2pk: float = Field(
        description="除P2PK外的量子计算易受攻击的比特币数量（单位：kBTC）",
        example=3636.0623877620596
    )
    p2sh_unknown: float = Field(
        description="未知赎回脚本的P2SH地址持有的比特币数量（单位：kBTC）",
        example=182.95050323096
    )
    p2pkh_hidden: float = Field(
        description="尚未暴露公钥的P2PKH地址持有的比特币数量（单位：kBTC）",
        example=7691.470119155041
    )
    p2sh_hidden: float = Field(
        description="已知赎回脚本的P2SH地址持有的比特币数量（单位：kBTC）",
        example=2314.8207187882203
    )
    lost: float = Field(
        description="估计永久丢失的比特币数量（单位：kBTC）",
        example=71.32284051986001
    )

    class Config:
        schema_extra = {
            "example": {
                "p2pk_total": 1763.15081330363,
                "quantum_vulnerable_minus_p2pk": 3636.0623877620596,
                "p2sh_unknown": 182.95050323096,
                "p2pkh_hidden": 7691.470119155041,
                "p2sh_hidden": 2314.8207187882203,
                "lost": 71.32284051986001
            }
        }

MAX_HEIGHT = 508000
SAMPLE_INTERVAL = 3000  # Sample every 3000 blocks

async def analyze_block_data(session, max_height=MAX_HEIGHT):
    """
    Analyze blockchain data and return statistics about address types and balances
    """
    # Check if tables exist
    tables_exist = await check_tables_exist(session)
    if not tables_exist:
        # Return error if tables don't exist
        raise HTTPException(
            status_code=500, 
            detail="Database tables not found. Please ensure the blockchain data is properly imported."
        )
    
    try:
        # Get blocks at sampling intervals
        blocks_query = select(Block.block_hash, Block.timestamp).order_by(Block.timestamp)
        blocks_result = await session.execute(blocks_query)
        blocks = blocks_result.all()
        
        # Prepare data structures for sampling at intervals
        sample_blocks = []
        current_height = 0
        next_sample = SAMPLE_INTERVAL
        
        for block_hash, timestamp in blocks:
            current_height += 1
            if current_height >= next_sample and current_height <= max_height:
                sample_blocks.append((current_height, block_hash, timestamp))
                next_sample += SAMPLE_INTERVAL
        
        # Prepare result containers
        heights = []
        timestamps = []
        total_values = []
        p2pk_values = []
        revealed_values = []
        potential_revealed_values = []
        
        # For each sample point, analyze the UTXO set
        for height, block_hash, timestamp in sample_blocks:
            # Get all UTXOs up to this block
            utxo_query = text("""
                SELECT o.txid, o.output_index, o.value, o.script_pub_key 
                FROM tx_outputs o
                LEFT JOIN tx_inputs i ON o.txid = i.prev_txid AND o.output_index = i.prev_output_index
                JOIN transactions tx ON o.txid = tx.txid
                JOIN blocks b ON tx.block_hash = b.block_hash
                WHERE b.timestamp <= :timestamp AND i.id IS NULL
            """)
            
            utxo_result = await session.execute(utxo_query, {"timestamp": timestamp})
            utxos = utxo_result.all()
            
            # Analyze UTXOs
            total_value = 0
            p2pk_value = 0
            revealed_value = 0
            p2sh_unknown_value = 0
            
            # Process each UTXO to determine its type and value
            for txid, output_index, value, script_pub_key in utxos:
                total_value += value
                
                # Interpret the locking script
                addr_type, _, _, _ = interpret_lock_script(script_pub_key)
                
                # Check if this is a P2PK type address
                if addr_type in [ADDR_P2PK, ADDR_P2PK_comp, ADDR_MULTISIG]:
                    p2pk_value += value
                    revealed_value += value
                
                # Check if public key has been revealed through spending
                elif addr_type == ADDR_P2PKH:
                    # Check if this address has ever been used as input (revealing the pubkey)
                    pubkey_revealed_query = text("""
                        SELECT COUNT(*) FROM tx_inputs i
                        JOIN tx_outputs o ON i.prev_txid = o.txid AND i.prev_output_index = o.output_index
                        WHERE o.script_pub_key = :script_pub_key AND i.script_sig IS NOT NULL
                        AND EXISTS (
                            SELECT 1 FROM transactions tx 
                            JOIN blocks b ON tx.block_hash = b.block_hash
                            WHERE tx.txid = i.txid AND b.timestamp <= :timestamp
                        )
                    """)
                    
                    result = await session.execute(pubkey_revealed_query, 
                                                {"script_pub_key": script_pub_key, "timestamp": timestamp})
                    count = result.scalar()
                    
                    if count > 0:
                        revealed_value += value
                
                # Check if this is P2SH with unknown script
                elif addr_type == ADDR_P2SH:
                    p2sh_unknown_value += value
            
            # Add data points
            heights.append(height)
            timestamps.append(timestamp)
            total_values.append(total_value * 1e-11)  # Convert satoshis to kBTC
            p2pk_values.append(p2pk_value * 1e-11)
            revealed_values.append(revealed_value * 1e-11)
            potential_revealed_values.append((revealed_value + p2sh_unknown_value) * 1e-11)
        
        return {
            "heights": heights,
            "timestamps": timestamps,
            "total_values": total_values,
            "p2pk_values": p2pk_values,
            "revealed_values": revealed_values,
            "potential_revealed_values": potential_revealed_values
        }
    except Exception as e:
        logger.error(f"Error analyzing block data: {e}")
        # Propagate the error
        raise HTTPException(
            status_code=500,
            detail=f"Database query error: {str(e)}"
        )

async def analyze_current_state(session):
    """
    Analyze the current state of the blockchain for address summary
    """
    # Check if tables exist
    tables_exist = await check_tables_exist(session)
    if not tables_exist:
        # Return error if tables don't exist
        raise HTTPException(
            status_code=500, 
            detail="Database tables not found. Please ensure the blockchain data is properly imported."
        )
    
    try:
        # Get latest block
        latest_block_query = select(Block.block_hash, Block.timestamp).order_by(desc(Block.timestamp)).limit(1)
        latest_block_result = await session.execute(latest_block_query)
        latest_block = latest_block_result.first()
        
        if not latest_block:
            raise HTTPException(status_code=404, detail="No block data available")
        
        latest_block_hash, latest_timestamp = latest_block
        
        # Get all current UTXOs
        utxo_query = text("""
            SELECT o.txid, o.output_index, o.value, o.script_pub_key 
            FROM tx_outputs o
            LEFT JOIN tx_inputs i ON o.txid = i.prev_txid AND o.output_index = i.prev_output_index
            WHERE i.id IS NULL
        """)
        
        utxo_result = await session.execute(utxo_query)
        utxos = utxo_result.all()
        
        # Initialize counters
        total_value = 0
        p2pk_value = 0
        revealed_value = 0
        p2sh_unknown_value = 0
        p2pkh_value = 0
        p2sh_value = 0
        lost_value = 0
        
        # Process each UTXO
        for txid, output_index, value, script_pub_key in utxos:
            total_value += value
            
            try:
                # Interpret the locking script
                addr_type, _, _, _ = interpret_lock_script(script_pub_key)
                
                # Categorize by type
                if addr_type in [ADDR_P2PK, ADDR_P2PK_comp, ADDR_MULTISIG]:
                    p2pk_value += value
                    revealed_value += value
                elif addr_type == ADDR_P2PKH:
                    p2pkh_value += value
                    
                    # Check if this address has ever been used as input (revealing the pubkey)
                    pubkey_revealed_query = text("""
                        SELECT COUNT(*) FROM tx_inputs i
                        JOIN tx_outputs o ON i.prev_txid = o.txid AND i.prev_output_index = o.output_index
                        WHERE o.script_pub_key = :script_pub_key AND i.script_sig IS NOT NULL
                    """)
                    
                    result = await session.execute(pubkey_revealed_query, {"script_pub_key": script_pub_key})
                    count = result.scalar()
                    
                    if count > 0:
                        revealed_value += value
                elif addr_type == ADDR_P2SH:
                    p2sh_value += value
                elif addr_type == ADDR_P2SH_OTHER:
                    p2sh_unknown_value += value
                elif addr_type == ADDR_UNKNOWN:
                    lost_value += value
            except Exception as e:
                # Count as lost value if we can't interpret the script
                lost_value += value
                logger.error(f"Error interpreting script: {e}")
        
        # Calculate the values
        p2pkh_hidden_value = p2pkh_value - (revealed_value - p2pk_value)
        
        # Convert to kBTC
        return {
            "p2pk_total": p2pk_value * 1e-11,
            "quantum_vulnerable_minus_p2pk": (revealed_value - p2pk_value) * 1e-11,
            "p2sh_unknown": p2sh_unknown_value * 1e-11,
            "p2pkh_hidden": p2pkh_hidden_value * 1e-11,
            "p2sh_hidden": p2sh_value * 1e-11,
            "lost": lost_value * 1e-11
        }
    except HTTPException:
        # Re-raise HTTP exceptions directly
        raise
    except Exception as e:
        logger.error(f"Error analyzing current state: {e}")
        # Propagate the error
        raise HTTPException(
            status_code=500,
            detail=f"Database query error: {str(e)}"
        )

@app.get(
    "/api/address-history",
    summary="获取比特币地址类型分布的历史数据（图表）",
    response_description="返回图表图片"
)
async def get_address_history_image():
    try:
        async with async_session() as session:
            # Analyze blockchain data
            history_data = await analyze_block_data(session)
            
            if not history_data["heights"]:
                raise HTTPException(status_code=404, detail="No block data available")
            
            # Using matplotlib to draw the plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(history_data["heights"], history_data["total_values"], label="Total Value (kBTC)")
            ax.plot(history_data["heights"], history_data["p2pk_values"], label="P2PK Value (kBTC)")
            ax.plot(history_data["heights"], history_data["revealed_values"], label="Revealed Value (kBTC)")
            ax.plot(history_data["heights"], history_data["potential_revealed_values"], 
                   label="Potential Revealed Value (kBTC)")
            
            ax.set_xlabel("Block Height")
            ax.set_ylabel("Value (kBTC)")
            ax.set_title("Bitcoin Address History")
            ax.legend()
            
            # Save plot to memory
            buf = BytesIO()
            fig.savefig(buf, format="png")
            plt.close(fig)
            buf.seek(0)
            
            return Response(content=buf.getvalue(), media_type="image/png")
    
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise e
    except Exception as e:
        logger.error(f"Error in /api/address-history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get(
    "/api/address-summary",
    summary="获取当前比特币地址类型分布摘要（图表）",
    response_description="返回图表图片"
)
async def get_address_summary_image():
    try:
        async with async_session() as session:
            # Get current state analysis
            summary_data = await analyze_current_state(session)
            
            # Using matplotlib to create a bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            categories = list(summary_data.keys())
            values = list(summary_data.values())
            
            ax.bar(categories, values)
            ax.set_ylabel("Value (kBTC)")
            ax.set_title("Bitcoin Address Summary")
            ax.tick_params(axis="x", rotation=45)
            
            # Save plot to memory
            buf = BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            
            return Response(content=buf.getvalue(), media_type="image/png")
    
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise e
    except Exception as e:
        logger.error(f"Error in /api/address-summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """
    API Root, return welcome message.
    """
    return {"message": "Welcome to Bitcoin Address Analysis API"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)