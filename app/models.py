from sqlalchemy import Column, Integer, String, TIMESTAMP, BigInteger, Float, ForeignKey, LargeBinary, Boolean
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Address(Base):
    __tablename__ = "address"
    
    address = Column(String(50), primary_key=True)
    address_type = Column(String(20), nullable=False)
    total_received = Column(BigInteger, default=0, nullable=False)
    total_sent = Column(BigInteger, default=0, nullable=False)
    balance = Column(BigInteger, default=0, nullable=False)
    pubkey_revealed = Column(Boolean, default=False, nullable=False)
    first_seen_block = Column(String(64))
    last_seen_block = Column(String(64))

class Block(Base):
    __tablename__ = "blocks"
    
    block_hash = Column(String(64), primary_key=True)
    version = Column(Integer, nullable=False)
    prev_block_hash = Column(String(64), nullable=False)
    merkle_root = Column(String(64), nullable=False)
    timestamp = Column(Integer, nullable=False)
    bits = Column(Integer, nullable=False)
    nonce = Column(Integer, nullable=False)
    block_size = Column(Integer, nullable=False)
    tx_count = Column(Integer, nullable=False)
    raw_block = Column(LargeBinary, nullable=False)
    file_name = Column(String(50))
    file_offset = Column(BigInteger)

class Transaction(Base):
    __tablename__ = "transactions"
    
    txid = Column(String(64), primary_key=True)
    block_hash = Column(String(64), ForeignKey('blocks.block_hash'), nullable=False)
    version = Column(Integer, nullable=False)
    input_count = Column(Integer, nullable=False)
    output_count = Column(Integer, nullable=False)
    lock_time = Column(Integer, nullable=False)
    raw_tx = Column(LargeBinary, nullable=False)

class TxInput(Base):
    __tablename__ = "tx_inputs"
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    txid = Column(String(64), ForeignKey('transactions.txid'), nullable=False)
    input_index = Column(Integer, nullable=False)
    prev_txid = Column(String(64), nullable=False)
    prev_output_index = Column(Integer, nullable=False)
    script_sig = Column(LargeBinary, nullable=False)
    sequence = Column(BigInteger, nullable=False)

class TxOutput(Base):
    __tablename__ = "tx_outputs"
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    txid = Column(String(64), ForeignKey('transactions.txid'), nullable=False)
    output_index = Column(Integer, nullable=False)
    value = Column(BigInteger, nullable=False)
    script_pub_key = Column(LargeBinary, nullable=False)

# class Snapshot(Base):
#     __tablename__ = "snapshots"
    
#     height = Column(Integer, primary_key=True)
#     snap_date = Column(TIMESTAMP)
#     tot_val = Column(BigInteger)
#     op_return = Column(BigInteger)
#     unknown = Column(BigInteger)
#     qattack_frac = Column(Float)
#     unknown_frac = Column(Float)

# class SnapshotSummaryByType(Base):
#     __tablename__ = "snapshot_summary_by_type"
    
#     snap_height = Column(Integer, ForeignKey('snapshots.height', ondelete='CASCADE'), primary_key=True)
#     addr_type = Column(String(32), primary_key=True)
#     num_pos = Column(Integer)
#     tot_val = Column(BigInteger)

# class SnapshotQuantumByType(Base):
#     __tablename__ = "snapshot_quantum_by_type"
    
#     snap_height = Column(Integer, ForeignKey('snapshots.height', ondelete='CASCADE'), primary_key=True)
#     addr_type = Column(String(32), primary_key=True)
#     num_pos = Column(Integer)
#     tot_val = Column(BigInteger) 