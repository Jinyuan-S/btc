# -- Database schema for Bitcoin blockchain data

# CREATE DATABASE btc_analysis;
# USE btc_analysis;

# -- Blocks table to store block header information
# CREATE TABLE blocks (
#     block_height BIGINT PRIMARY KEY,
#     block_hash CHAR(64) NOT NULL UNIQUE,
#     previous_block_hash CHAR(64) NOT NULL,
#     merkle_root CHAR(64) NOT NULL,
#     timestamp BIGINT NOT NULL,
#     bits INT UNSIGNED NOT NULL,
#     nonce INT UNSIGNED NOT NULL,
#     version INT NOT NULL,
#     size INT NOT NULL,
#     transaction_count INT NOT NULL,
#     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#     INDEX idx_timestamp (timestamp),
#     INDEX idx_prev_hash (previous_block_hash)
# );

# -- Transactions table to store transaction information
# CREATE TABLE transactions (
#     tx_id CHAR(64) PRIMARY KEY,
#     block_height BIGINT NOT NULL,
#     version INT NOT NULL,
#     locktime INT UNSIGNED NOT NULL,
#     size INT NOT NULL,
#     weight INT NOT NULL,
#     is_coinbase BOOLEAN NOT NULL,
#     is_segwit BOOLEAN NOT NULL,
#     input_count INT NOT NULL,
#     output_count INT NOT NULL,
#     total_input_value BIGINT NOT NULL DEFAULT 0,
#     total_output_value BIGINT NOT NULL DEFAULT 0,
#     fee BIGINT NOT NULL DEFAULT 0,
#     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#     FOREIGN KEY (block_height) REFERENCES blocks(block_height),
#     INDEX idx_block_height (block_height)
# );

# -- Transaction inputs table
# CREATE TABLE transaction_inputs (
#     id BIGINT AUTO_INCREMENT PRIMARY KEY,
#     tx_id CHAR(64) NOT NULL,
#     input_index INT NOT NULL,
#     prev_tx_id CHAR(64),
#     prev_output_index INT,
#     script_sig TEXT NOT NULL,
#     sequence INT UNSIGNED NOT NULL,
#     witness TEXT,
#     address VARCHAR(100),
#     value BIGINT,
#     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#     FOREIGN KEY (tx_id) REFERENCES transactions(tx_id),
#     INDEX idx_address (address),
#     INDEX idx_prev_tx (prev_tx_id, prev_output_index)
# );

# -- Transaction outputs table
# CREATE TABLE transaction_outputs (
#     id BIGINT AUTO_INCREMENT PRIMARY KEY,
#     tx_id CHAR(64) NOT NULL,
#     output_index INT NOT NULL,
#     value BIGINT NOT NULL,
#     script_pubkey TEXT NOT NULL,
#     address VARCHAR(100),
#     address_type VARCHAR(20),
#     is_spent BOOLEAN DEFAULT FALSE,
#     spent_by_tx_id CHAR(64),
#     spent_at_block BIGINT,
#     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#     FOREIGN KEY (tx_id) REFERENCES transactions(tx_id),
#     FOREIGN KEY (spent_at_block) REFERENCES blocks(block_height),
#     INDEX idx_address (address),
#     INDEX idx_unspent (is_spent),
#     INDEX idx_spent_by (spent_by_tx_id)
# );

# -- Addresses table to store address statistics
# CREATE TABLE addresses (
#     address VARCHAR(100) PRIMARY KEY,
#     first_seen_block BIGINT NOT NULL,
#     last_seen_block BIGINT NOT NULL,
#     total_received BIGINT NOT NULL DEFAULT 0,
#     total_sent BIGINT NOT NULL DEFAULT 0,
#     balance BIGINT NOT NULL DEFAULT 0,
#     transaction_count INT NOT NULL DEFAULT 0,
#     address_type VARCHAR(20),
#     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#     updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
#     FOREIGN KEY (first_seen_block) REFERENCES blocks(block_height),
#     FOREIGN KEY (last_seen_block) REFERENCES blocks(block_height),
#     INDEX idx_balance (balance),
#     INDEX idx_type (address_type)
# );


import mysql.connector
from mysql.connector import Error
from blockchain_parser.blockchain import Blockchain
import hashlib
import base58
from datetime import datetime
import configparser
from typing import Dict, List, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class BTCMySQLParser:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.conn = None
        self.cursor = None
        self.blockchain = Blockchain(self.config['blockchain']['dat_path'])
        self.batch_size = int(self.config['processing']['batch_size'])
        
    def _load_config(self, config_path: str) -> configparser.ConfigParser:
        config = configparser.ConfigParser()
        config.read(config_path)
        return config

    def connect_db(self):
        try:
            self.conn = mysql.connector.connect(
                host=self.config['mysql']['host'],
                user=self.config['mysql']['user'],
                password=self.config['mysql']['password'],
                database=self.config['mysql']['database']
            )
            self.cursor = self.conn.cursor()
            logging.info("Successfully connected to MySQL database")
        except Error as e:
            logging.error(f"Error connecting to MySQL: {e}")
            raise

    def parse_blocks(self):
        blocks_batch = []
        txs_batch = []
        inputs_batch = []
        outputs_batch = []
        address_updates = {}

        for block in self.blockchain.get_ordered_blocks():
            block_data = self._parse_block(block)
            blocks_batch.append(block_data)

            for tx in block.transactions:
                tx_data = self._parse_transaction(tx, block.height)
                txs_batch.append(tx_data)
                
                inputs_batch.extend(self._parse_inputs(tx))
                outputs_batch.extend(self._parse_outputs(tx))

            if len(blocks_batch) >= self.batch_size:
                self._insert_batch_data(blocks_batch, txs_batch, inputs_batch, outputs_batch)
                self._update_addresses(address_updates)
                blocks_batch = []
                txs_batch = []
                inputs_batch = []
                outputs_batch = []
                address_updates = {}

    def _parse_block(self, block) -> tuple:
        return (
            block.height,
            block.hash,
            block.header.previous_block_hash,
            block.header.merkle_root,
            block.header.timestamp,
            block.header.bits,
            block.header.nonce,
            block.header.version,
            block.size,
            len(block.transactions)
        )

    def _parse_transaction(self, tx, block_height: int) -> tuple:
        return (
            tx.hash,
            block_height,
            tx.version,
            tx.locktime,
            tx.size,
            tx.weight,
            tx.is_coinbase(),
            tx.is_segwit(),
            len(tx.inputs),
            len(tx.outputs),
            sum(inp.value for inp in tx.inputs if inp.value is not None),
            sum(out.value for out in tx.outputs),
            self._calculate_fee(tx)
        )

    def _parse_inputs(self, tx) -> List[tuple]:
        inputs = []
        for idx, inp in enumerate(tx.inputs):
            address = self._get_address_from_script(inp.script)
            inputs.append((
                tx.hash,
                idx,
                inp.transaction_hash,
                inp.transaction_index,
                inp.script.hex(),
                inp.sequence,
                inp.witness.hex() if tx.is_segwit() else None,
                address,
                inp.value
            ))
        return inputs

    def _parse_outputs(self, tx) -> List[tuple]:
        outputs = []
        for idx, out in enumerate(tx.outputs):
            address, addr_type = self._get_address_and_type_from_script(out.script)
            outputs.append((
                tx.hash,
                idx,
                out.value,
                out.script.hex(),
                address,
                addr_type,
                False,  # is_spent
                None,  # spent_by_tx_id
                None   # spent_at_block
            ))
        return outputs

    def _insert_batch_data(self, blocks: List[tuple], txs: List[tuple], 
                          inputs: List[tuple], outputs: List[tuple]):
        try:
            # Insert blocks
            self.cursor.executemany("""
                INSERT INTO blocks (block_height, block_hash, previous_block_hash, 
                merkle_root, timestamp, bits, nonce, version, size, transaction_count)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, blocks)

            # Insert transactions
            self.cursor.executemany("""
                INSERT INTO transactions (tx_id, block_height, version, locktime, 
                size, weight, is_coinbase, is_segwit, input_count, output_count,
                total_input_value, total_output_value, fee)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, txs)

            # Insert inputs
            self.cursor.executemany("""
                INSERT INTO transaction_inputs (tx_id, input_index, prev_tx_id,
                prev_output_index, script_sig, sequence, witness, address, value)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, inputs)

            # Insert outputs
            self.cursor.executemany("""
                INSERT INTO transaction_outputs (tx_id, output_index, value,
                script_pubkey, address, address_type, is_spent, spent_by_tx_id,
                spent_at_block)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, outputs)

            self.conn.commit()
            logging.info(f"Inserted batch: {len(blocks)} blocks, {len(txs)} transactions")
        except Error as e:
            self.conn.rollback()
            logging.error(f"Error inserting batch: {e}")
            raise

    def _calculate_fee(self, tx) -> int:
        if tx.is_coinbase():
            return 0
        return sum(inp.value for inp in tx.inputs if inp.value is not None) - \
               sum(out.value for out in tx.outputs)

    def _get_address_from_script(self, script) -> str:
        try:
            return script.get_address()
        except:
            return None

    def _get_address_and_type_from_script(self, script) -> Tuple[str, str]:
        try:
            address = script.get_address()
            script_type = script.type
            return address, script_type
        except:
            return None, None

    def close(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            logging.info("Database connection closed")

if __name__ == "__main__":
    # Example configuration file (config.ini):
    # [mysql]
    # host = localhost
    # user = your_username
    # password = your_password
    # database = btc_analysis
    #
    # [blockchain]
    # dat_path = /path/to/bitcoin/blocks
    #
    # [processing]
    # batch_size = 1000

    parser = BTCMySQLParser('config.ini')
    try:
        parser.connect_db()
        parser.parse_blocks()
    finally:
        parser.close()