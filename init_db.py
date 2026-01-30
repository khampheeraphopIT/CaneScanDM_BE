import asyncio
from src.database import engine, Base
from src.models import UsageStats, PredictionHistory

async def init_db():
    print("⏳ Initializing database tables in Supabase...")
    async with engine.begin() as conn:
        # Create all tables defined in models.py
        await conn.run_sync(Base.metadata.create_all)
    print("✅ Database initialization complete!")

if __name__ == "__main__":
    asyncio.run(init_db())
