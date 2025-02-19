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

