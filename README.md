# API endpoint

Before running, add `.env` file under BTC root, add setting to database.

```
# MySQL配置
MYSQL_USER=
MYSQL_PASSWORD=
MYSQL_HOST=
MYSQL_PORT=
MYSQL_DB=
```

## How to run backend

```bash
python -m app.main
```

## UI 文档
`http://localhost:8000/docs`: 交互式swagger ui doc
`http://localhost:8000/redoc`: ReDoc文档

## 1. 地址历史数据接口

url: `http://localhost:8000/api/address-history` 绘制address history表
method: `GET`
description: 获取比特币地址类型分布的历史数据，包括总量、P2PK地址、已知公钥地址等数据随区块高度的变化。

response type:
```typescript
interface AddressHistoryResponse {
    height: number[];           // 区块高度数组
    total_value: number[];      // 对应高度的总币量（单位：kBTC）
    p2pk_value: number[];      // P2PK类型地址持有的币量（单位：kBTC）
    revealed_value: number[];   // 已公开公钥的地址持有的币量（单位：kBTC）
    pot_revealed_value: number[]; // 潜在暴露的币量（包括已知公钥和P2SH unknown，单位：kBTC）
}
```


## 2. 地址分布摘要饼图接口

url: `http://localhost:8000/api/address-summary` 绘制address summary表
method: `GET`
description: 获取最新区块的比特币地址类型分布摘要，包括不同类型地址持有的比特币数量及其安全性分类。

response type: 
```typescript
interface AddressSummaryResponse {
    p2pk_total: number;                    // P2PK地址总量（包括multisig和compressed）
    quantum_vulnerable_minus_p2pk: number;  // 量子易受攻击（除P2PK外）的比特币数量
    p2sh_unknown: number;                  // 未知脚本的P2SH地址持有量
    p2pkh_hidden: number;                  // 未暴露公钥的P2PKH地址持有量
    p2sh_hidden: number;                   // 已知脚本的P2SH地址持有量
    lost: number;                          // 估计丢失的比特币数量
}
```

## 3. 修改了后端部分代码，实现以二进制向前端发送图片

## 4. 新的数据库设计文件(详细看下面的NEW Data Structure)和改进的dat2mysql.py文件已上传，目前已经优化到约20分钟插入一个完整的dat，采用多线程（默认4线程）分析dat到临时csv，然后用事务批量提交到数据库中，保证每个dat的数据都能完整写入到数据库。此外，程序支持随时中断，每次插入完一个完整的dat后，终端会提示用户是否继续，用户可以在5秒内输入非Y字符来正常结束程序；如果输入Y或没有任何输入，5秒之后程序继续，直到项目根目录下的/dat目录中所有dat被写入完毕。所有的临时csv都会在每个dat成功写入mysql后被清除，即使用户手动终止了程序，也会清除临时csv来释放硬盘空间。checkpoint会保存下一个要执行的dat文件，每次执行程序都会从这里开始，所以不要轻易修改或删除它。如果需要重新开始写入，需清空address, tx_inputs, tx_outputs, transactions, blocks表以及清空checkpoint.json中的全部内容（推荐使用以下命令快速清除大表数据）
### a. use btc_analysis;
### b.  SET FOREIGN_KEY_CHECKS = 0;
### c. TRUNCATE TABLE address;
### d. TRUNCATE TABLE tx_inputs;
### e. TRUNCATE TABLE tx_outputs;
### f. TRUNCATE TABLE transactions;
### g. TRUNCATE TABLE blocks;
### h. SET FOREIGN_KEY_CHECKS = 1;

## 5. 执行脚本前，需要手动修改mysql的配置，windows可以找到my.ini配置文件，在[mysqld]的下面输入 local_infile=1 ，然后保存配置文件，重新启动mysql服务，这样可以允许mysql通过读取csv来批量插入数据。（默认是不允许的，所以要手动改配置文件）

## 6. 执行dat2mysql.py脚本后，如果遇到类似“The total number of locks exceeds the lock table size”的错误，在mysql终端中输入以下命令：
### a. show variables like "%_buffer%";
### b. SET GLOBAL innodb_buffer_pool_size=67108864;   // 可以改为更大的值，这个是3x1024x1024x1024的大小



# NEW Data Structure

```sql
CREATE TABLE `address`  (
  `address` varchar(50) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL,
  `address_type` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL,
  `total_received` bigint NOT NULL DEFAULT 0,
  `total_sent` bigint NOT NULL DEFAULT 0,
  `balance` bigint NOT NULL DEFAULT 0,
  `pubkey_revealed` tinyint(1) NOT NULL DEFAULT 0,
  `first_seen_block` char(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `last_seen_block` char(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  PRIMARY KEY (`address`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;

CREATE TABLE `blocks`  (
  `block_hash` char(64) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL,
  `version` int NOT NULL,
  `prev_block_hash` char(64) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL,
  `merkle_root` char(64) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL,
  `timestamp` int NOT NULL,
  `bits` int NOT NULL,
  `nonce` int NOT NULL,
  `block_size` int NOT NULL,
  `tx_count` int NOT NULL,
  `raw_block` longblob NOT NULL,
  `file_name` varchar(50) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `file_offset` bigint NULL DEFAULT NULL,
  PRIMARY KEY (`block_hash`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;

CREATE TABLE `transactions`  (
  `txid` char(64) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL,
  `block_hash` char(64) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL,
  `version` int NOT NULL,
  `input_count` int NOT NULL,
  `output_count` int NOT NULL,
  `lock_time` int NOT NULL,
  `raw_tx` longblob NOT NULL,
  PRIMARY KEY (`txid`) USING BTREE,
  INDEX `idx_block_hash`(`block_hash` ASC) USING BTREE,
  CONSTRAINT `fk_tx_block` FOREIGN KEY (`block_hash`) REFERENCES `blocks` (`block_hash`) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE = InnoDB CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;

DROP TABLE IF EXISTS `tx_inputs`;
CREATE TABLE `tx_inputs`  (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `txid` char(64) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL,
  `input_index` int NOT NULL,
  `prev_txid` char(64) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL,
  `prev_output_index` int NOT NULL,
  `script_sig` longblob NOT NULL,
  `sequence` bigint NOT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  INDEX `idx_txid`(`txid` ASC) USING BTREE,
  CONSTRAINT `fk_input_tx` FOREIGN KEY (`txid`) REFERENCES `transactions` (`txid`) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE = InnoDB CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;

CREATE TABLE `tx_outputs`  (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `txid` char(64) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL,
  `output_index` int NOT NULL,
  `value` bigint NOT NULL,
  `script_pub_key` longblob NOT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  INDEX `idx_txid`(`txid` ASC) USING BTREE,
  CONSTRAINT `fk_output_tx` FOREIGN KEY (`txid`) REFERENCES `transactions` (`txid`) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE = InnoDB CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;


```





# Old Data Structure

```sql
CREATE TABLE addresses (
  id            INT AUTO_INCREMENT PRIMARY KEY,  -- 自增主键(仅为内部引用)
  keyhash       VARCHAR(64) NOT NULL,            -- 或 BINARY(32) / VARBINARY / HEX ... 公钥hash
  addr          VARCHAR(64),                     -- base58地址字符串
  type          TINYINT,                         -- 地址类型 (1=P2PK向公钥支付,2=P2PK_comp压缩公钥,10=P2PKH,20=P2SH...)
  val           BIGINT,                          -- 当前余额 (satoshi)
  key_seen      INT DEFAULT 0,                   -- 见过几次公钥
  ins_count     INT DEFAULT 0,                   -- 传入次数(可选)
  outs_count    INT DEFAULT 0,                   -- 传出次数(可选)
  last_height   INT,                             -- 最后使用区块高度(可选)
  UNIQUE (keyhash)
);

CREATE TABLE snapshots (
  height       INT PRIMARY KEY,    -- 与 _id 相同,块高度
  snap_date    DATETIME,           -- 对应 doc['date']，该快照对应的区块生成时间
  tot_val      BIGINT,            -- doc['tot-val'], 全网在链btc余额
  op_return    BIGINT,            -- doc['op-return']被op-return销毁的btc总量
  unknown      BIGINT,            -- doc['unknown']无法识别或者丢失的btc总量
  qattack_frac DOUBLE,            -- doc['qattack-frac']量子可攻击的份额（0，1）
  unknown_frac DOUBLE             -- doc['unknown-frac']unkonwn占全网比例
  -- 还可以加更多顶层字段 ...
);

CREATE TABLE snapshot_quantum_by_type (
  snap_height   INT,
  addr_type     VARCHAR(32),
  num_pos       INT,            --已经见过公钥大于0的地址数
  tot_val       BIGINT,         --量子脆弱地址总余额
  PRIMARY KEY (snap_height, addr_type),
  CONSTRAINT fk_snapheight2
    FOREIGN KEY (snap_height)
    REFERENCES snapshots (height)
    ON DELETE CASCADE
);

CREATE TABLE snapshot_summary_by_type (
  snap_height   INT,                -- 外键指向 snapshots.height
  addr_type     VARCHAR(32),        -- 'P2PKH', 'P2SH', 'P2PK'...
  num_pos       INT,                -- doc['summary-by-type'][].num-pos 此类型地址余额大于零的地址数量
  tot_val       BIGINT,             -- doc['summary-by-type'][].tot-val 此类型地址总余额
  PRIMARY KEY (snap_height, addr_type),
  CONSTRAINT fk_snapheight
    FOREIGN KEY (snap_height) 
    REFERENCES snapshots (height)
    ON DELETE CASCADE
);


```


