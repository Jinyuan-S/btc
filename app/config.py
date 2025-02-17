from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MONGO_URL: str = "mongodb://localhost:27017"
    MYSQL_URL: str = "mysql+asyncmy://user:pass@localhost/dbname"
    
settings = Settings()