from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
import os
from dotenv import load_dotenv

load_dotenv()

# Get DB URL from env
DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL:
    # Ensure it uses asyncpg
    if DATABASE_URL.startswith("postgresql://"):
        DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
    
    # Supabase/Render fix: Handle special characters and ensure SSL
    # Note: asyncpg sometimes needs specific SSL args in the engine creator
else:
    DATABASE_URL = "postgresql+asyncpg://postgres:postgres@localhost/postgres"

# Build engine with extra protection for production
engine = create_async_engine(
    DATABASE_URL, 
    echo=False,
    pool_pre_ping=True,  # Check connection validity before using
    pool_recycle=300,    # Restart connections every 5 mins to prevent stale links
    connect_args={
        "command_timeout": 60,  # Increase timeout for commands
        "server_settings": {
            "tcp_user_timeout": "30000" # 30 seconds TCP timeout
        }
    }
)
SessionLocal = async_sessionmaker(autocommit=False, autoflush=False, bind=engine, class_=AsyncSession)

class Base(DeclarativeBase):
    pass

async def get_db():
    async with SessionLocal() as session:
        yield session
