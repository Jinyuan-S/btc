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
from fastapi import Response, HTTPException
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sqlalchemy import select, and_

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Create async engine with connection pool settings
    engine = create_async_engine(
        settings.MYSQL_URL,
        echo=True,  # Set to True to see SQL statements
        pool_size=5,  # Connection pool size
        max_overflow=10,  # Max connections that can be created beyond pool_size
        pool_timeout=30,  # Timeout for getting connection from pool
        pool_recycle=1800,  # Time after which connections are recycled
        pool_pre_ping=True,  # Check if connection is valid before using
        connect_args={
            "connect_timeout": 10,  # Connection timeout
            "charset": "utf8mb4",
            "use_unicode": True,
            "autocommit": False
        }
    )
except Exception as e:
    logger.error(f"Failed to create engine: {e}")
    raise  # Re-raise the exception to prevent the app from starting with a broken database connection

try:
    # Create async session factory
    async_session = sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False
    )
except Exception as e:
    logger.error(f"Failed to create sessionmaker: {e}")
    raise  # Re-raise the exception to prevent the app from starting with a broken session factory

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
    Bitcoin Address Analysis API providing historical data and current summary information on address type distribution.
    
    Main features:
    * Get historical data on Bitcoin address type distribution
    * Get current Bitcoin address type distribution summary
    * Analyze potential impact of quantum computing on the Bitcoin network
    
    All amount data is in kBTC (thousand Bitcoin).
    """,
    version="1.0.0",
    contact={
        "name": "Your Name",
        "email": "your.email@example.com",
    },
)

class AddressHistoryResponse(BaseModel):
    """
    Bitcoin address historical data response model
    """
    height: List[int] = Field(
        description="List of block heights, one sampling point per 3000 blocks",
        example=[3000, 6000, 9000]
    )
    total_value: List[float] = Field(
        description="Total circulating Bitcoin supply at corresponding block heights (unit: kBTC)",
        example=[150.0, 300.0, 450.0]
    )
    p2pk_value: List[float] = Field(
        description="Bitcoin amount held by P2PK type addresses (including multisig and compressed) (unit: kBTC)",
        example=[146.02308, 288.97308, 429.37275]
    )
    revealed_value: List[float] = Field(
        description="Bitcoin amount held by addresses with known public keys (quantum vulnerable part) (unit: kBTC)",
        example=[146.08699, 289.03699, 429.43666]
    )
    pot_revealed_value: List[float] = Field(
        description="Potentially exposed Bitcoin amount (including known public keys and P2SH unknown) (unit: kBTC)",
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
    Bitcoin address distribution summary response model
    """
    p2pk_total: float = Field(
        description="Total Bitcoin amount held by P2PK type addresses (including multisig and compressed) (unit: kBTC)",
        example=1763.15081330363
    )
    quantum_vulnerable_minus_p2pk: float = Field(
        description="Bitcoin amount vulnerable to quantum computing excluding P2PK (unit: kBTC)",
        example=3636.0623877620596
    )
    p2sh_unknown: float = Field(
        description="Bitcoin amount held by P2SH addresses with unknown redemption scripts (unit: kBTC)",
        example=182.95050323096
    )
    p2pkh_hidden: float = Field(
        description="Bitcoin amount held by P2PKH addresses with unexposed public keys (unit: kBTC)",
        example=7691.470119155041
    )
    p2sh_hidden: float = Field(
        description="Bitcoin amount held by P2SH addresses with known redemption scripts (unit: kBTC)",
        example=2314.8207187882203
    )
    lost: float = Field(
        description="Estimated permanently lost Bitcoin amount (unit: kBTC)",
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

@app.get(
    "/api/address-history",
    summary="Get historical data of Bitcoin address type distribution (chart)",
    response_description="Returns chart image"
)
async def get_address_history_image():
    try:
        async with async_session() as session:
            # Query all snapshot data
            query = select(Snapshot).where(Snapshot.height <= MAX_HEIGHT).order_by(Snapshot.height)
            result = await session.execute(query)
            snapshots = result.scalars().all()

            if not snapshots:
                raise HTTPException(status_code=404, detail="No snapshot data available")

            height = []
            total_value = []
            p2pk_value = []
            revealed_value = []
            pot_revealed_value = []

            for snapshot in snapshots:
                h = snapshot.height
                height.append(h)

                tot = float(snapshot.tot_val) * 1e-8 * 1e-3
                total_value.append(tot)

                # Get P2PK related data
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

                # Calculate quantum vulnerable amount
                qval = snapshot.qattack_frac * tot if snapshot.qattack_frac is not None else 0
                revealed_value.append(qval)

                # Get P2SH unknown data
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

            # Create plot with matplotlib
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(height, total_value, label="Total Value (kBTC)")
            ax.plot(height, p2pk_value, label="P2PK Value (kBTC)")
            ax.set_xlabel("Block Height")
            ax.set_ylabel("Value (kBTC)")
            ax.set_title("Bitcoin Address History")
            ax.legend()

            # Save chart to memory
            buf = BytesIO()
            fig.savefig(buf, format="png")
            plt.close(fig)
            buf.seek(0)

            return Response(content=buf.getvalue(), media_type="image/png")
    except Exception as e:
        logger.error(f"Error in /api/address-history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get(
    "/api/address-summary",
    summary="Get current summary of Bitcoin address type distribution (chart)",
    response_description="Returns chart image"
)
async def get_address_summary_image():
    try:
        async with async_session() as session:
            # Get latest snapshot
            query = select(Snapshot).where(Snapshot.height <= MAX_HEIGHT).order_by(Snapshot.height.desc()).limit(1)
            result = await session.execute(query)
            latest_snapshot = result.scalar_one()

            height = latest_snapshot.height
            tot = float(latest_snapshot.tot_val) * 1e-8 * 1e-3
            lost = latest_snapshot.unknown_frac * tot
            qval = latest_snapshot.qattack_frac * tot

            # Get P2PK related data
            p2pk_query = select(SnapshotSummaryByType).where(
                and_(
                    SnapshotSummaryByType.snap_height == height,
                    SnapshotSummaryByType.addr_type.in_(['P2PK', 'P2PK multisig', 'P2PK comp'])
                )
            )
            p2pk_result = await session.execute(p2pk_query)
            p2pk_data = p2pk_result.scalars().all()
            valp2pk = sum(float(d.tot_val) * 1e-8 * 1e-3 for d in p2pk_data)

            # Get P2PKH data
            p2pkh_query = select(SnapshotSummaryByType).where(
                and_(
                    SnapshotSummaryByType.snap_height == height,
                    SnapshotSummaryByType.addr_type == 'P2PKH'
                )
            )
            p2pkh_result = await session.execute(p2pkh_query)
            p2pkh = p2pkh_result.scalar_one()
            finalp2pkh = float(p2pkh.tot_val) * 1e-8 * 1e-3

            # Get P2SH data (including P2SH and P2SH unknown)
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
                else:  # 'P2SH unknown'
                    valp2shu = float(item.tot_val) * 1e-8 * 1e-3

            # Create summary data dictionary for chart
            summary_data = {
                "P2PK Total": valp2pk,
                "Quantum Vulnerable (non-P2PK)": qval - valp2pk,
                "P2SH Unknown": valp2shu,
                "P2PKH Hidden": finalp2pkh - qval + valp2pk,
                "P2SH Hidden": valp2sh,
                "Lost": lost
            }

        # Create bar chart with matplotlib
        fig, ax = plt.subplots(figsize=(10, 6))
        categories = list(summary_data.keys())
        values = list(summary_data.values())
        ax.bar(categories, values)
        ax.set_ylabel("Value (kBTC)")
        ax.set_title("Bitcoin Address Summary")
        ax.tick_params(axis="x", rotation=45)

        # Save chart to memory
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)

        return Response(content=buf.getvalue(), media_type="image/png")
    except Exception as e:
        logger.error(f"Error in /api/address-summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

class AddressInfoResponse(BaseModel):
    """
    Bitcoin address information response model
    """
    address: str = Field(
        description="Bitcoin address",
        example="1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"
    )
    address_type: str = Field(
        description="Address type (P2PK, P2PK_comp, P2PKH, P2SH etc.)",
        example="P2PKH"
    )
    balance: int = Field(
        description="Current balance (unit: satoshi)",
        example=50000000
    )
    key_seen: int = Field(
        description="Public key exposure count",
        example=1
    )
    ins_count: int = Field(
        description="Received transaction count",
        example=10
    )
    outs_count: int = Field(
        description="Sent transaction count",
        example=5
    )
    last_height: Optional[int] = Field(
        description="Last block height used",
        example=700000,
        default=None
    )

    class Config:
        schema_extra = {
            "example": {
                "address": "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
                "address_type": "P2PKH",
                "balance": 50000000,
                "key_seen": 1,
                "ins_count": 10,
                "outs_count": 5,
                "last_height": 700000
            }
        }

def get_address_type(type_int: int) -> str:
    """Convert address type integer to string representation"""
    type_map = {
        1: "P2PK",
        2: "P2PK_comp",
        10: "P2PKH",
        20: "P2SH"
    }
    return type_map.get(type_int, "unknown")

@app.get(
    "/api/address/{address}",
    summary="Get detailed information for a specific Bitcoin address",
    response_model=AddressInfoResponse,
    response_description="Returns detailed information about the address, including balance, transaction history, etc."
)
async def get_address_info(address: str):
    """
    Get detailed information for a specific Bitcoin address, including:
    * Address type
    * Current balance
    * Public key exposure count
    * Transaction count statistics
    * Last block height used
    """
    try:
        async with async_session() as session:
            # Add debug logging
            logger.info(f"Searching for address: {address}")

            # Add timeout and limit to the query
            query = select(Address).where(
                (Address.addr == address) | (Address.keyhash == address)
            ).limit(1)  # Limit to 1 result

            try:
                # Add timeout to the execute call
                result = await session.execute(query)
                address_info = result.scalar_one_or_none()
            except Exception as db_error:
                logger.error(f"Database query error: {db_error}", exc_info=True)
                raise HTTPException(
                    status_code=500,
                    detail="Database query timeout or error"
                )

            if not address_info:
                logger.warning(f"Address not found: {address}")
                raise HTTPException(
                    status_code=404,
                    detail=f"Address {address} not found"
                )

            logger.info(f"Found address info: {address_info.addr}, type: {address_info.type}, balance: {address_info.val}")

            return AddressInfoResponse(
                address=address_info.addr or address_info.keyhash,  # Use keyhash if addr is None
                address_type=get_address_type(address_info.type),
                balance=address_info.val or 0,  # Handle None values
                key_seen=address_info.key_seen or 0,
                ins_count=address_info.ins_count or 0,
                outs_count=address_info.outs_count or 0,
                last_height=address_info.last_height  # Now accepts None
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /api/address/{address}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

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