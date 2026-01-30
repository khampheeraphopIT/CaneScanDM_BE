from sqlalchemy import Column, Integer, Float, Boolean, Date, DateTime
from sqlalchemy.sql import func
from .base import Base

class UsageStats(Base):
    __tablename__ = "usage_stats"

    id = Column(Integer, primary_key=True, index=True)
    daily_usage_count = Column(Integer, default=0)
    last_reset_date = Column(Date, index=True) # Format: YYYY-MM-DD
    is_daily_quota_full = Column(Boolean, default=False)
    manual_wait_until = Column(Float, nullable=True) # Timestamp
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), default=func.now())
