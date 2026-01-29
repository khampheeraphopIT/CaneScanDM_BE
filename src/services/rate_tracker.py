import time
from collections import deque
from typing import Optional, Dict, Any
from datetime import datetime, timedelta, timezone

class RateLimitTracker:
    """Track API usage locally with specific support for daily quotas"""
    
    def __init__(self, max_per_minute: int = 5, max_per_day: int = 20):
        self.max_per_minute = max_per_minute
        self.max_per_day = max_per_day
        self.requests = deque()  # Store timestamps
        self._last_minute_requests = deque()
        self._last_day_requests = deque()
        self._manual_wait_until: Optional[float] = None
        self._is_daily_quota_full: bool = False

    def _cleanup(self):
        now = time.time()
        # Minute cleanup
        while self._last_minute_requests and self._last_minute_requests[0] < now - 60:
            self._last_minute_requests.popleft()
        # Day cleanup (optional if we use fixed reset, but good for rolling)
        while self._last_day_requests and self._last_day_requests[0] < now - 86400:
            self._last_day_requests.popleft()

    def record_request(self):
        now = time.time()
        self._last_minute_requests.append(now)
        self._last_day_requests.append(now)
        self._cleanup()

    def set_rate_limited(self, retry_after: int):
        """Set manual wait time from API response (usually minute level)"""
        self._manual_wait_until = time.time() + retry_after

    def set_daily_quota_exceeded(self):
        """Specifically handle 429 RESOURCE_EXHAUSTED for the day"""
        self._is_daily_quota_full = True

    def get_seconds_until_midnight_utc(self) -> int:
        """Calculate seconds until UTC midnight (approximate Google Quota reset)"""
        now = datetime.now(timezone.utc)
        tomorrow = now + timedelta(days=1)
        reset_time = tomorrow.replace(hour=0, minute=0, second=0, microsecond=0)
        return int((reset_time - now).total_seconds())

    def to_dict(self) -> Dict[str, Any]:
        self._cleanup()
        now = time.time()
        
        # Requests used
        used_minute = len(self._last_minute_requests)
        used_day = len(self._last_day_requests)
        
        # Calculate next available
        next_available_in = 0
        if self._manual_wait_until and self._manual_wait_until > now:
            next_available_in = int(self._manual_wait_until - now)
        elif used_minute >= self.max_per_minute:
            # Wait for the oldest request in the last minute to expire
            next_available_in = int(max(1, 60 - (now - self._last_minute_requests[0])))

        # Daily quota logic
        if self._is_daily_quota_full or used_day >= self.max_per_day:
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
