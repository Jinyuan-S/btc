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




# Data structure

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

last time accessed.
fist time accessed.