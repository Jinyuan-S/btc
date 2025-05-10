# Bitcoin Address Analysis Tool

A comprehensive Bitcoin blockchain data analysis system that provides insights on address types, quantum vulnerability, and distribution patterns across the blockchain.


## Guidelines for Starting the Project
1. If you are running this project on another computer, make sure that all required dependencies and software are installed and configured. (Python3.8.10, NodeJS v22.14.0, VS Code/IntelliJ Idea, Navicat Premium 17, MySQL8.0(and create the four tables listed below), MongoDB Compass, etc.)
2. Download the latest .dat block file to the specified directory on this server. Directory: `E:\Prof Marco FYP Blockchain\data\blocks` 
(PS: The last block file in the directory is blk03480.dat. Please make sure that the next file to download is blk03481.dat. The files in the directory must be consecutive and there can be no duplicates.)
3. Once the block data file has been downloaded to the directory, launch the dat2mongo.py script. The script will continue to write data to MongoDB, depending on where it left off, until it gets to the last file in the directory. 
(PS: You can also interrupt the program when prompted to do so.)
4. When the preceding program shows that all blocks have been written to MongoDB, you can launch the mongo2MySQL.py script. This will import all the data from MongoDB into MySQL.
(PS: The script also supports progress reads.)
5. Once you've finished running the scripts, you can run analyzed.py to generate the graphs that for this project.
6. Open the terminal with administrator privileges in the project root and type `python -m app.main` to start the backend.
7. If the backend was started successfully, open a new terminal with administrator privileges and enter `npm run dev` in the `/bitcoin_front_end` directory. Once the frontend has started successfully, the console will print the web address of the frontend, for example: http://localhost:5173/
8. At this point, the project has been successfully deployed to the server. If you want to access the project elsewhere, you first need to link to NUS VPN and then type the web address (Format: http://ServerIP:5173/, for example, http://10.248.8.247:5173/) into your browser to access it.

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


## Database Structure

### Current Schema

```sql
DROP TABLE IF EXISTS `addresses`;
CREATE TABLE `addresses`  (
    `id` int NOT NULL AUTO_INCREMENT,
    `keyhash` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
    `addr` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL,
    `type` tinyint NULL DEFAULT NULL,
    `val` bigint NULL DEFAULT NULL,
    `key_seen` int NULL DEFAULT 0,
    `ins_count` int NULL DEFAULT 0,
    `outs_count` int NULL DEFAULT 0,
    `last_height` int NULL DEFAULT NULL,
    PRIMARY KEY (`id`) USING BTREE,
    UNIQUE INDEX `keyhash`(`keyhash` ASC) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

DROP TABLE IF EXISTS `snapshots`;
CREATE TABLE `snapshots`  (
                              `height` int NOT NULL,
                              `snap_date` datetime NULL DEFAULT NULL,
                              `tot_val` bigint NULL DEFAULT NULL,
                              `op_return` bigint NULL DEFAULT NULL,
                              `unknown` bigint NULL DEFAULT NULL,
                              `qattack_frac` double NULL DEFAULT NULL,
                              `unknown_frac` double NULL DEFAULT NULL,
                              PRIMARY KEY (`height`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

DROP TABLE IF EXISTS `snapshot_quantum_by_type`;
CREATE TABLE `snapshot_quantum_by_type`  (
    `snap_height` int NOT NULL,
    `addr_type` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
    `num_pos` int NULL DEFAULT NULL,
    `tot_val` bigint NULL DEFAULT NULL,
    PRIMARY KEY (`snap_height`, `addr_type`) USING BTREE,
    CONSTRAINT `fk_snapheight2` FOREIGN KEY (`snap_height`) REFERENCES `snapshots` (`height`) ON DELETE CASCADE ON UPDATE RESTRICT
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

DROP TABLE IF EXISTS `snapshot_summary_by_type`;
CREATE TABLE `snapshot_summary_by_type`  (
    `snap_height` int NOT NULL,
    `addr_type` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
    `num_pos` int NULL DEFAULT NULL,
    `tot_val` bigint NULL DEFAULT NULL,
    PRIMARY KEY (`snap_height`, `addr_type`) USING BTREE,
    CONSTRAINT `fk_snapheight` FOREIGN KEY (`snap_height`) REFERENCES `snapshots` (`height`) ON DELETE CASCADE ON UPDATE RESTRICT
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

```
