import asyncio
import aiohttp
import time
from typing import Dict, Any, List
from datetime import datetime
import motor.motor_asyncio
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm

# docker-compose up -d

class BitcoinDataAPI:
    def __init__(self):
        self.blockstream_url = "https://blockstream.info/api"
        self.session = None
        # Use Motor for async MongoDB operations
        self.mongo_client = motor.motor_asyncio.AsyncIOMotorClient("mongodb://admin:password123@localhost:27017")
        self.db = self.mongo_client["btc"]
        self.blocks_collection = self.db["blocks"]
        self.transactions_collection = self.db["transactions"]
        
    async def _init_session(self):
        if self.session is None or self.session.closed:  # Add check for closed session
            self.session = aiohttp.ClientSession()
    
    async def _make_request(self, url: str, max_retries: int = 3) -> Dict[str, Any]:
        """异步请求方法，带重试机制"""
        await self._init_session()
        for attempt in range(max_retries):
            try:
                async with self.session.get(url) as response:
                    if response.status == 429:  # Rate limit hit
                        wait_time = int(response.headers.get('Retry-After', 60))
                        print(f"Rate limit hit, waiting {wait_time} seconds")
                        await asyncio.sleep(wait_time)
                        continue
                        
                    response.raise_for_status()
                    if 'application/json' in response.headers.get('Content-Type', ''):
                        return await response.json()
                    return await response.text()
            except aiohttp.ClientError as e:
                if attempt == max_retries - 1:
                    print(f"Failed after {max_retries} attempts: {e}")
                    raise
                wait_time = 2 ** attempt
                print(f"Request failed, retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)

    async def get_block(self, height: int) -> Dict[str, Any]:
        """异步获取完整区块数据"""
        # 获取区块hash
        hash_url = f"{self.blockstream_url}/block-height/{height}"
        block_hash = await self._make_request(hash_url)
        
        # 并发获取区块信息和首批交易
        block_url = f"{self.blockstream_url}/block/{block_hash}"
        try:
            block_data, first_txs = await asyncio.gather(
                self._make_request(block_url),
                self._make_request(f"{self.blockstream_url}/block/{block_hash}/txs/0")
            )
        except Exception as e:
            print(f"Error fetching block data for height {height}: {e}")
            raise

        # 获取所有交易（并发处理分页）
        all_transactions = list(first_txs) if isinstance(first_txs, list) else []  # Add type check
        tasks = []
        start_index = 25
        
        while len(all_transactions) < block_data.get('tx_count', float('inf')):  # Add tx count check
            try:
                txs_url = f"{self.blockstream_url}/block/{block_hash}/txs/{start_index}"
                tasks.append(self._make_request(txs_url))
                start_index += 25
                if len(tasks) >= 10:  # 每批处理10个并发请求
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    tasks = []  # Clear tasks before processing results
                    for result in results:
                        if isinstance(result, Exception):
                            print(f"Error fetching transactions at index {start_index}: {result}")
                            continue
                        if not result:
                            return self._build_block_data(block_hash, height, block_data, all_transactions)
                        all_transactions.extend(result)
            except Exception as e:
                print(f"Error in transaction batch at index {start_index}: {e}")
                break
        
        if tasks:  # 处理剩余的请求
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, list):
                    all_transactions.extend(result)
        
        return self._build_block_data(block_hash, height, block_data, all_transactions)
    
    def _build_block_data(self, block_hash, height, block_data, transactions):
        """构建区块数据结构"""
        return {
            'hash': block_hash,
            'height': height,
            'version': block_data['version'],
            'merkleroot': block_data['merkle_root'],
            'time': block_data['timestamp'],
            'nonce': block_data['nonce'],
            'bits': block_data['bits'],
            'difficulty': block_data['difficulty'],
            'previousblockhash': block_data['previousblockhash'],
            'tx': transactions,
            'size': block_data['size'],
            'weight': block_data['weight'],
            'tx_count': len(transactions)
        }

    async def get_transaction_details(self, txid: str) -> Dict[str, Any]:
        """异步获取交易详情"""
        url = f"{self.blockstream_url}/tx/{txid}"
        tx_data = await self._make_request(url)
        
        # 并发获取所有输入的前置交易
        input_tasks = []
        for vin in tx_data['vin']:
            if 'txid' in vin and vin['txid'] != '0' * 64:
                prev_tx_url = f"{self.blockstream_url}/tx/{vin['txid']}"
                input_tasks.append((vin, self._make_request(prev_tx_url)))
        
        if input_tasks:
            for vin, task in input_tasks:
                try:
                    prev_tx = await task
                    vin['prev_output'] = prev_tx['vout'][vin['vout']]
                except Exception:
                    pass
        
        return tx_data

    async def store_block(self, block_data: Dict[str, Any]) -> None:
        """异步存储区块数据到MongoDB"""
        try:
            # 使用update_one with upsert来避免重复插入
            await self.blocks_collection.update_one(
                {"hash": block_data["hash"]},
                {"$set": block_data},
                upsert=True
            )
        except Exception as e:
            print(f"Error storing block {block_data['height']}: {e}")

    async def store_transaction(self, tx_data: Dict[str, Any]) -> None:
        """异步存储交易数据到MongoDB"""
        try:
            await self.transactions_collection.update_one(
                {"txid": tx_data["txid"]},
                {"$set": tx_data},
                upsert=True
            )
        except Exception as e:
            print(f"Error storing transaction {tx_data['txid']}: {e}")

    async def sync_blocks_range(self, start_height: int, end_height: int, 
                              batch_size: int = 10,
                              process_count: int = None) -> None:
        """使用多进程和异步IO同步指定范围的区块到数据库"""
        if process_count is None:
            process_count = cpu_count()
            
        # Main progress bar for overall block range
        total_blocks = end_height - start_height + 1
        with tqdm(total=total_blocks, desc="Overall Progress", unit="blocks") as pbar:
            for batch_start in range(start_height, end_height + 1, batch_size):
                batch_end = min(batch_start + batch_size, end_height + 1)
                tasks = []
                
                try:
                    # 并发获取区块数据
                    for height in range(batch_start, batch_end):
                        tasks.append(self.get_block(height))
                    blocks = await asyncio.gather(*tasks)
                    
                    # 并发存储区块和交易数据
                    store_tasks = []
                    # Progress bar for transactions in current batch
                    total_txs = sum(len(block['tx']) for block in blocks)
                    with tqdm(total=total_txs, desc=f"Processing txs for blocks {batch_start}-{batch_end-1}", 
                            unit="tx", leave=False) as tx_pbar:
                        for block in blocks:
                            # 存储区块
                            store_tasks.append(self.store_block(block))
                            
                            # 获取并存储交易
                            for tx in block['tx']:
                                tx_detail = await self.get_transaction_details(tx['txid'])
                                store_tasks.append(self.store_transaction(tx_detail))
                                tx_pbar.update(1)
                    
                    # 等待所有存储操作完成
                    await asyncio.gather(*store_tasks)
                    pbar.update(batch_end - batch_start)
                    
                    # 短暂暂停避免请求过快
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    print(f"Error syncing blocks {batch_start}-{batch_end-1}: {e}")

    async def close(self):
        """关闭会话"""
        if self.session:
            await self.session.close()
            self.session = None


async def main():
    api = BitcoinDataAPI()
    try:
        int_start_height = 820000
        int_end_height = 820010
        print("Start sync from height: ", int_start_height, " to height: ", int_end_height)
        
        time_start = time.time()
        # 同步一个范围的区块
        await api.sync_blocks_range(
            start_height=int_start_height,
            end_height=int_end_height,
            batch_size=5
        )
        time_end = time.time()
        print(f"\nTime taken: {time_end - time_start} seconds")
    except Exception as e:
        print(f"Error in main: {e}")
        raise
    finally:
        await api.close()
        await api.mongo_client.close()  # Add await for proper MongoDB cleanup

if __name__ == "__main__":
    asyncio.run(main())
