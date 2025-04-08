from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # MySQL配置
    MYSQL_USER: str = "root"
    MYSQL_PASSWORD: str = "gsj123"
    MYSQL_HOST: str = "localhost"
    MYSQL_PORT: int = 3306
    MYSQL_DB: str = "btc_3"
    
    @property
    def MYSQL_URL(self) -> str:
        return f"mysql+aiomysql://{self.MYSQL_USER}:{self.MYSQL_PASSWORD}@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DB}"
    
    # MongoDB配置（如果需要的话）
    MONGO_URL: Optional[str] = None

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = Settings()