import time
import asyncio
from typing import Optional, Dict, Any
from datetime import datetime, timedelta, timezone
from sqlalchemy import select, update
from src.config.database import SessionLocal
from src.models.usage import UsageStats

class RateLimitTracker:
    """Track API usage with PostgreSQL persistence and UTC midnight resets"""
    
    def __init__(self, max_per_minute: int = 5, max_per_day: int = 20):
        self.max_per_minute = max_per_minute
        self.max_per_day = max_per_day
        
        # We still keep minute tracking in RAM (fast cleanup, not critical to survive restarts)
        self._last_minute_requests = []
        self._manual_wait_until: Optional[float] = None

    def _get_current_utc_date(self) -> datetime.date:
        return datetime.now(timezone.utc).date()

    async def _get_or_create_stats(self, session):
        """Fetch stats for today or create a new row if date changed"""
        today = self._get_current_utc_date()
        result = await session.execute(select(UsageStats).where(UsageStats.last_reset_date == today))
        stats = result.scalars().first()
        
        if not stats:
            # Check if there's any old stats and delete or just create new
            stats = UsageStats(daily_usage_count=0, last_reset_date=today, is_daily_quota_full=False)
            session.add(stats)
            await session.commit()
            await session.refresh(stats)
        return stats

    def _cleanup_minute(self):
        now = time.time()
        self._last_minute_requests = [t for t in self._last_minute_requests if t > now - 60]

    async def record_request(self):
        async with SessionLocal() as session:
            stats = await self._get_or_create_stats(session)
            stats.daily_usage_count += 1
            await session.commit()
            
            # RAM part
            now = time.time()
            self._last_minute_requests.append(now)
            self._cleanup_minute()

    async def set_rate_limited(self, retry_after: int):
        """Set manual wait time (usually minute level)"""
        async with SessionLocal() as session:
            stats = await self._get_or_create_stats(session)
            stats.manual_wait_until = time.time() + retry_after
            await session.commit()
        
        self._manual_wait_until = time.time() + retry_after

    async def set_daily_quota_exceeded(self):
        """Specifically handle 429 RESOURCE_EXHAUSTED for the day"""
        async with SessionLocal() as session:
            stats = await self._get_or_create_stats(session)
            stats.is_daily_quota_full = True
            # Ensure count is at least the max
            if stats.daily_usage_count < self.max_per_day:
                stats.daily_usage_count = self.max_per_day
            await session.commit()

    def get_seconds_until_midnight_utc(self) -> int:
        now = datetime.now(timezone.utc)
        tomorrow = now + timedelta(days=1)
        reset_time = tomorrow.replace(hour=0, minute=0, second=0, microsecond=0)
        return int((reset_time - now).total_seconds())

    async def to_dict(self) -> Dict[str, Any]:
        async with SessionLocal() as session:
            stats = await self._get_or_create_stats(session)
            
            self._cleanup_minute()
            now = time.time()
            
            used_minute = len(self._last_minute_requests)
            used_day = stats.daily_usage_count
            
            # Calculate next available
            next_available_in = 0
            
            # Manual wait (highest priority)
            db_manual_wait = stats.manual_wait_until or 0
            if db_manual_wait > now:
                next_available_in = int(db_manual_wait - now)
            elif self._manual_wait_until and self._manual_wait_until > now:
                next_available_in = int(self._manual_wait_until - now)
                
            # Minute limit
            elif used_minute >= self.max_per_minute:
                wait_time = int(max(1, 60 - (now - self._last_minute_requests[0])))
                next_available_in = max(next_available_in, wait_time)

            # Daily limit
            is_limit_reached = stats.is_daily_quota_full or used_day >= self.max_per_day
            
            if is_limit_reached:
                daily_reset_in = self.get_seconds_until_midnight_utc()
                next_available_in = max(next_available_in, daily_reset_in)
                can_request = False
                status_code = "daily_limit"
            else:
                can_request = next_available_in <= 0
                status_code = "ok" if can_request else "minute_limit"

            return {
                "requests_used_minute": used_minute,
                "requests_used_day": used_day,
                "max_per_minute": self.max_per_minute,
                "max_per_day": self.max_per_day,
                "remaining_minute": max(0, self.max_per_minute - used_minute),
                "remaining_day": max(0, self.max_per_day - used_day),
                "next_available_in": next_available_in,
                "can_request": can_request,
                "status_code": status_code
            }

# Global instance
rate_tracker = RateLimitTracker()
