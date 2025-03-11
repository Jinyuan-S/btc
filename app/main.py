from fastapi import FastAPI, HTTPException
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, and_
from app.config import settings
from app.models import Snapshot, SnapshotSummaryByType, SnapshotQuantumByType, Address
from contextlib import asynccontextmanager
from typing import List, Optional
from pydantic import BaseModel, Field
import logging
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # 创建异步引擎，设置连接池
    engine = create_async_engine(
        settings.MYSQL_URL,
    echo=True,  # 设置为True可以看到SQL语句
    pool_size=5,  # 连接池大小
    max_overflow=10,  # 超过pool_size后最多可以创建的连接数
    pool_timeout=30,  # 连接池获取连接的超时时间
        pool_recycle=1800,  # 连接在连接池中重用的时间，超过后会被回收
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

class AddressResponse(BaseModel):
    """
    比特币地址信息响应模型
    """
    keyhash: str = Field(
        description="公钥哈希",
        example="1a2b3c4d..."
    )
    addr: str = Field(
        description="Base58格式的比特币地址",
        example="1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"
    )
    type: int = Field(
        description="地址类型: 1=P2PK, 2=P2PK_comp, 10=P2PKH, 20=P2SH",
        example=10
    )
    val: int = Field(
        description="当前余额(单位: satoshi)",
        example=5000000000
    )
    key_seen: int = Field(
        description="公钥出现次数",
        example=3
    )
    ins_count: int = Field(
        description="接收交易次数",
        example=10
    )
    outs_count: int = Field(
        description="发送交易次数",
        example=5
    )
    last_height: Optional[int] = Field(
        description="最后活动的区块高度",
        example=500000
    )

    class Config:
        schema_extra = {
            "example": {
                "keyhash": "1a2b3c4d...",
                "addr": "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
                "type": 10,
                "val": 5000000000,
                "key_seen": 3,
                "ins_count": 10,
                "outs_count": 5,
                "last_height": 500000
            }
        }

MAX_HEIGHT = 508000

@app.get(
    "/api/address-history",
    response_model=AddressHistoryResponse,
    summary="获取比特币地址类型分布的历史数据",
    description="""
    返回比特币地址类型分布的历史数据，包括：
    * 区块高度（每3000个区块一个采样点）
    * 总流通量
    * P2PK地址持有量
    * 已知公钥地址持有量
    * 潜在暴露地址持有量
    
    所有金额均以kBTC（千比特币）为单位。
    数据最大区块高度限制为508000。
    """,
    response_description="包含地址类型分布历史数据的响应对象"
)
async def get_address_history():
    """
    获取比特币地址类型分布的历史数据。
    
    返回从创世区块开始，每3000个区块的采样数据，包括不同类型地址的持有量变化。
    """
    try:
        async with async_session() as session:
            # 获取所有快照数据
            query = select(Snapshot).where(Snapshot.height <= MAX_HEIGHT).order_by(Snapshot.height)
            result = await session.execute(query)
            snapshots = result.scalars().all()
            
            height = []
            total_value = []
            p2pk_value = []
            revealed_value = []
            pot_revealed_value = []
            
            for snapshot in snapshots:
                h = snapshot.height
                height.append(h)
                
                # 转换为kBTC
                tot = float(snapshot.tot_val) * 1e-8 * 1e-3
                total_value.append(tot)
                
                # 获取P2PK相关数据
                p2pk_query = select(SnapshotSummaryByType).where(
                    and_(
                        SnapshotSummaryByType.snap_height == h,
                        SnapshotSummaryByType.addr_type.in_(['P2PK', 'P2PK multisig', 'P2PK comp'])
                    )
                )
                p2pk_result = await session.execute(p2pk_query)
                p2pk_data = p2pk_result.scalars().all()
                
                valp2pk = sum(float(d.tot_val) * 1e-8 * 1e-3 for d in p2pk_data)
                p2pk_value.append(valp2pk)
                
                # 计算量子易受攻击的金额
                qval = snapshot.qattack_frac * tot
                revealed_value.append(qval)
                
                # 获取P2SH unknown数据
                p2sh_query = select(SnapshotSummaryByType).where(
                    and_(
                        SnapshotSummaryByType.snap_height == h,
                        SnapshotSummaryByType.addr_type == 'P2SH unknown'
                    )
                )
                p2sh_result = await session.execute(p2sh_query)
                p2sh_unknown = p2sh_result.scalar()
                valp2shu = float(p2sh_unknown.tot_val) * 1e-8 * 1e-3 if p2sh_unknown else 0
                
                pot_revealed_value.append(qval + valp2shu)
            
            return AddressHistoryResponse(
                height=height,
                total_value=total_value,
                p2pk_value=p2pk_value,
                revealed_value=revealed_value,
                pot_revealed_value=pot_revealed_value
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get(
    "/api/address-summary",
    response_model=AddressSummaryResponse,
    summary="获取当前比特币地址类型分布摘要",
    description="""
    返回最新区块的比特币地址类型分布摘要，包括：
    * P2PK地址总量
    * 量子易受攻击地址（非P2PK）持有量
    * 未知脚本的P2SH地址持有量
    * 未暴露公钥的P2PKH地址持有量
    * 已知脚本的P2SH地址持有量
    * 估计丢失的比特币数量
    
    所有金额均以kBTC（千比特币）为单位。
    """,
    response_description="包含当前地址类型分布摘要的响应对象"
)
async def get_address_summary():
    """
    获取当前比特币地址类型分布的摘要信息。
    
    返回最新区块中各类型地址的持有量及其安全性分类。
    """
    try:
        async with async_session() as session:
            # 获取最新快照
            query = select(Snapshot).where(Snapshot.height <= MAX_HEIGHT).order_by(Snapshot.height.desc()).limit(1)
            result = await session.execute(query)
            latest_snapshot = result.scalar_one()
            
            height = latest_snapshot.height
            tot = float(latest_snapshot.tot_val) * 1e-8 * 1e-3
            lost = latest_snapshot.unknown_frac * tot
            qval = latest_snapshot.qattack_frac * tot
            
            # 获取P2PK相关数据
            p2pk_query = select(SnapshotSummaryByType).where(
                and_(
                    SnapshotSummaryByType.snap_height == height,
                    SnapshotSummaryByType.addr_type.in_(['P2PK', 'P2PK multisig', 'P2PK comp'])
                )
            )
            p2pk_result = await session.execute(p2pk_query)
            p2pk_data = p2pk_result.scalars().all()
            valp2pk = sum(float(d.tot_val) * 1e-8 * 1e-3 for d in p2pk_data)
            
            # 获取P2PKH数据
            p2pkh_query = select(SnapshotSummaryByType).where(
                and_(
                    SnapshotSummaryByType.snap_height == height,
                    SnapshotSummaryByType.addr_type == 'P2PKH'
                )
            )
            p2pkh_result = await session.execute(p2pkh_query)
            p2pkh = p2pkh_result.scalar_one()
            finalp2pkh = float(p2pkh.tot_val) * 1e-8 * 1e-3
            
            # 获取P2SH数据
            p2sh_query = select(SnapshotSummaryByType).where(
                and_(
                    SnapshotSummaryByType.snap_height == height,
                    SnapshotSummaryByType.addr_type.in_(['P2SH', 'P2SH unknown'])
                )
            )
            p2sh_result = await session.execute(p2sh_query)
            p2sh_data = p2sh_result.scalars().all()
            
            valp2sh = 0
            valp2shu = 0
            for item in p2sh_data:
                if item.addr_type == 'P2SH':
                    valp2sh = float(item.tot_val) * 1e-8 * 1e-3
                else:  # P2SH unknown
                    valp2shu = float(item.tot_val) * 1e-8 * 1e-3
            
            return AddressSummaryResponse(
                p2pk_total=valp2pk,
                quantum_vulnerable_minus_p2pk=qval - valp2pk,
                p2sh_unknown=valp2shu,
                p2pkh_hidden=finalp2pkh - qval + valp2pk,
                p2sh_hidden=valp2sh,
                lost=lost
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get(
    "/api/address/{address}",
    response_model=AddressResponse,
    summary="获取比特币地址详细信息",
    description="""
    根据比特币地址查询详细信息，包括：
    * 公钥哈希
    * 地址类型
    * 当前余额
    * 公钥出现次数
    * 交易统计
    * 最后活动区块高度
    
    地址类型说明：
    * 1 = P2PK (向公钥支付)
    * 2 = P2PK_comp (压缩公钥)
    * 10 = P2PKH (向公钥哈希支付)
    * 20 = P2SH (向脚本哈希支付)
    """,
    response_description="包含地址详细信息的响应对象"
)
async def get_address_info(address: str):
    """
    获取指定比特币地址的详细信息。
    
    Args:
        address: Base58格式的比特币地址
        
    Returns:
        AddressResponse: 地址详细信息
        
    Raises:
        HTTPException: 当地址不存在或发生其他错误时
    """
    try:
        async with async_session() as session:
            query = select(Address).where(Address.addr == address)
            result = await session.execute(query)
            address_info = result.scalar_one_or_none()
            
            if not address_info:
                raise HTTPException(
                    status_code=404,
                    detail=f"Address {address} not found"
                )
            
            return AddressResponse(
                keyhash=address_info.keyhash,
                addr=address_info.addr,
                type=address_info.type,
                val=address_info.val,
                key_seen=address_info.key_seen,
                ins_count=address_info.ins_count,
                outs_count=address_info.outs_count,
                last_height=address_info.last_height
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error querying address {address}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error while querying address: {str(e)}"
        )

@app.get("/")
async def root():
    """
    API Root, return welcome message.
    """
    return {"message": "Welcome to Bitcoin Address Analysis API"}

# @app.get("/users")
# async def get_users():
#     try:
#         async with async_session() as session:
#             # real badskllhgkasdjf
#             result = await session.execute("SELECT * FROM users LIMIT 10")
#             users = result.fetchall()
#             return {"users": [dict(user) for user in users]}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/blocks")
# async def get_blocks():
#     blocks = await app.mongodb.blocks.find().to_list(length=100)
#     return blocks

# @app.get("/transactions")
# async def get_transactions():
#     transactions = await app.mongodb.transactions.find().to_list(length=100)

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)