# Bitcoin Address Analysis Tool

A comprehensive Bitcoin blockchain data analysis system that provides insights on address types, quantum vulnerability, and distribution patterns across the blockchain.

## Setup

Before running, create a `.env` file in the project root with your database configuration:

```
# MySQL Configuration
MYSQL_USER=
MYSQL_PASSWORD=
MYSQL_HOST=
MYSQL_PORT=
MYSQL_DB=
```

## API Documentation

Interactive API documentation is available at:
- `http://localhost:8000/docs`: Swagger UI
- `http://localhost:8000/redoc`: ReDoc

## API Endpoints

### 1. Address History Data

**URL**: `http://localhost:8000/api/address-history`  
**Method**: `GET`  
**Description**: Provides historical data on Bitcoin address type distribution across block heights, including total value, P2PK addresses, and addresses with revealed public keys.

**Response Type**:
```typescript
interface AddressHistoryResponse {
    height: number[];           // Block heights array
    total_value: number[];      // Total coin amount at each height (kBTC)
    p2pk_value: number[];       // Coins held in P2PK addresses (kBTC)
    revealed_value: number[];   // Coins in addresses with revealed public keys (kBTC)
    pot_revealed_value: number[]; // Potentially exposed coins (including known public keys and P2SH unknown, kBTC)
}
```

### 2. Address Distribution Summary

**URL**: `http://localhost:8000/api/address-summary`  
**Method**: `GET`  
**Description**: Provides a summary of Bitcoin address type distribution at the latest block, including the amount of Bitcoin held by different address types and their security classification.

**Response Type**:
```typescript
interface AddressSummaryResponse {
    p2pk_total: number;                    // Total P2PK addresses (including multisig and compressed)
    quantum_vulnerable_minus_p2pk: number;  // Quantum vulnerable Bitcoin (excluding P2PK)
    p2sh_unknown: number;                  // Bitcoin held in P2SH addresses with unknown scripts
    p2pkh_hidden: number;                  // Bitcoin held in P2PKH addresses with hidden public keys
    p2sh_hidden: number;                   // Bitcoin held in P2SH addresses with known scripts
    lost: number;                          // Estimated lost Bitcoin
}
```

### 3. Single Address Query

**URL**: `http://localhost:8000/api/address/{address}`  
**Method**: `GET`  
**Example**: `GET /api/address/1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa`

## Data Processing

The project includes a `dat2mysql.py` script that imports Bitcoin blockchain data (.dat files) into MySQL. The optimized implementation:

- Uses multi-threading (default 4 threads) to analyze .dat files into temporary CSV files
- Employs transaction batching for efficient database insertion
- Processes a complete .dat file in approximately 20 minutes
- Supports interruption and resumption via checkpoints
- Automatically cleans up temporary CSV files after successful imports

### Running the Import Script

1. Configure MySQL to allow local file imports:
   - Add `local_infile=1` to the `[mysqld]` section in your MySQL configuration file (e.g., my.ini on Windows)
   - Restart the MySQL service

2. If you encounter lock table size errors, increase the InnoDB buffer pool size:
   ```sql
   SET GLOBAL innodb_buffer_pool_size=67108864; -- Can be set to a larger value as needed
   ```

3. To restart the import process, clear existing tables:
   ```sql
   USE btc_analysis;
   SET FOREIGN_KEY_CHECKS = 0;
   TRUNCATE TABLE address;
   TRUNCATE TABLE tx_inputs;
   TRUNCATE TABLE tx_outputs;
   TRUNCATE TABLE transactions;
   TRUNCATE TABLE blocks;
   SET FOREIGN_KEY_CHECKS = 1;
   ```
   Also clear the contents of `checkpoint.json` to start from the beginning.

## Database Structure

### Current Schema

```sql
CREATE TABLE `address` (
  `address` varchar(50) NOT NULL,
  `address_type` varchar(20) NOT NULL,
  `total_received` bigint NOT NULL DEFAULT 0,
  `total_sent` bigint NOT NULL DEFAULT 0,
  `balance` bigint NOT NULL DEFAULT 0,
  `pubkey_revealed` tinyint(1) NOT NULL DEFAULT 0,
  `first_seen_block` char(64) DEFAULT NULL,
  `last_seen_block` char(64) DEFAULT NULL,
  PRIMARY KEY (`address`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 ROW_FORMAT=DYNAMIC;

CREATE TABLE `blocks` (
  `block_hash` char(64) NOT NULL,
  `version` int NOT NULL,
  `prev_block_hash` char(64) NOT NULL,
  `merkle_root` char(64) NOT NULL,
  `timestamp` int NOT NULL,
  `bits` int NOT NULL,
  `nonce` int NOT NULL,
  `block_size` int NOT NULL,
  `tx_count` int NOT NULL,
  `raw_block` longblob NOT NULL,
  `file_name` varchar(50) DEFAULT NULL,
  `file_offset` bigint DEFAULT NULL,
  PRIMARY KEY (`block_hash`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 ROW_FORMAT=DYNAMIC;

CREATE TABLE `transactions` (
  `txid` char(64) NOT NULL,
  `block_hash` char(64) NOT NULL,
  `version` int NOT NULL,
  `input_count` int NOT NULL,
  `output_count` int NOT NULL,
  `lock_time` int NOT NULL,
  `raw_tx` longblob NOT NULL,
  PRIMARY KEY (`txid`),
  KEY `idx_block_hash` (`block_hash`),
  CONSTRAINT `fk_tx_block` FOREIGN KEY (`block_hash`) REFERENCES `blocks` (`block_hash`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 ROW_FORMAT=DYNAMIC;

CREATE TABLE `tx_inputs` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `txid` char(64) NOT NULL,
  `input_index` int NOT NULL,
  `prev_txid` char(64) NOT NULL,
  `prev_output_index` int NOT NULL,
  `script_sig` longblob NOT NULL,
  `sequence` bigint NOT NULL,
  PRIMARY KEY (`id`),
  KEY `idx_txid` (`txid`),
  CONSTRAINT `fk_input_tx` FOREIGN KEY (`txid`) REFERENCES `transactions` (`txid`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 ROW_FORMAT=DYNAMIC;

CREATE TABLE `tx_outputs` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `txid` char(64) NOT NULL,
  `output_index` int NOT NULL,
  `value` bigint NOT NULL,
  `script_pub_key` longblob NOT NULL,
  PRIMARY KEY (`id`),
  KEY `idx_txid` (`txid`),
  CONSTRAINT `fk_output_tx` FOREIGN KEY (`txid`) REFERENCES `transactions` (`txid`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 ROW_FORMAT=DYNAMIC;
```
