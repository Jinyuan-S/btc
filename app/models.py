from sqlalchemy import Column, Integer, String, TIMESTAMP, BigInteger, Float, ForeignKey
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Snapshot(Base):
    __tablename__ = "snapshots"
    
    height = Column(Integer, primary_key=True)
    snap_date = Column(TIMESTAMP)
    tot_val = Column(BigInteger)
    op_return = Column(BigInteger)
    unknown = Column(BigInteger)
    qattack_frac = Column(Float)
    unknown_frac = Column(Float)

class SnapshotSummaryByType(Base):
    __tablename__ = "snapshot_summary_by_type"
    
    snap_height = Column(Integer, ForeignKey('snapshots.height', ondelete='CASCADE'), primary_key=True)
    addr_type = Column(String(32), primary_key=True)
    num_pos = Column(Integer)
    tot_val = Column(BigInteger)

class SnapshotQuantumByType(Base):
    __tablename__ = "snapshot_quantum_by_type"
    
    snap_height = Column(Integer, ForeignKey('snapshots.height', ondelete='CASCADE'), primary_key=True)
    addr_type = Column(String(32), primary_key=True)
    num_pos = Column(Integer)
    tot_val = Column(BigInteger) 