/*
 Navicat Premium Data Transfer

 Source Server         : MySQL
 Source Server Type    : MySQL
 Source Server Version : 80023
 Source Host           : localhost:3306
 Source Schema         : btc_analysis

 Target Server Type    : MySQL
 Target Server Version : 80023
 File Encoding         : 65001

 Date: 12/03/2025 11:42:16
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for address
-- ----------------------------
DROP TABLE IF EXISTS `address`;
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

-- ----------------------------
-- Table structure for blocks
-- ----------------------------
DROP TABLE IF EXISTS `blocks`;
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

-- ----------------------------
-- Table structure for transactions
-- ----------------------------
DROP TABLE IF EXISTS `transactions`;
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

-- ----------------------------
-- Table structure for tx_inputs
-- ----------------------------
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

-- ----------------------------
-- Table structure for tx_outputs
-- ----------------------------
DROP TABLE IF EXISTS `tx_outputs`;
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

SET FOREIGN_KEY_CHECKS = 1;
