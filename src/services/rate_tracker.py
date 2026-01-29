import time
import json
import os
from typing import Optional, Dict, Any
from datetime import datetime, timedelta, timezone

# Path for persistent storage
USAGE_FILE = "usage_stats.json"

class RateLimitTracker:
    """Track API usage locally with persistence and UTC midnight resets"""
    
    def __init__(self, max_per_minute: int = 5, max_per_day: int = 20):
        self.max_per_minute = max_per_minute
        self.max_per_day = max_per_day
        
        # State
        self._last_minute_requests = []
        self._daily_usage_count = 0
        self._last_reset_date = "" # Format: YYYY-MM-DD (UTC)
        self._manual_wait_until: Optional[float] = None
        self._is_daily_quota_full: bool = False
        
        self._load()
        self._check_daily_reset()

    def _get_current_utc_date(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _check_daily_reset(self):
        """Reset the day counter if the UTC date has changed"""
        current_date = self._get_current_utc_date()
        if self._last_reset_date != current_date:
            self._daily_usage_count = 0
            self._last_reset_date = current_date
            self._is_daily_quota_full = False
            self._save()

    def _load(self):
        """Load stats from file"""
        if os.path.exists(USAGE_FILE):
            try:
                with open(USAGE_FILE, "r") as f:
                    data = json.load(f)
                    self._daily_usage_count = data.get("daily_usage_count", 0)
                    self._last_reset_date = data.get("last_reset_date", "")
                    self._is_daily_quota_full = data.get("is_daily_quota_full", False)
                    self._manual_wait_until = data.get("manual_wait_until")
            except Exception as e:
                print(f"Error loading usage file: {e}")

    def _save(self):
        """Save stats to file"""
        try:
            with open(USAGE_FILE, "w") as f:
                json.dump({
                    "daily_usage_count": self._daily_usage_count,
                    "last_reset_date": self._last_reset_date,
                    "is_daily_quota_full": self._is_daily_quota_full,
                    "manual_wait_until": self._manual_wait_until,
                    "updated_at": time.time()
                }, f)
        except Exception as e:
            print(f"Error saving usage file: {e}")

    def _cleanup_minute(self):
        now = time.time()
        self._last_minute_requests = [t for t in self._last_minute_requests if t > now - 60]

    def record_request(self):
        self._check_daily_reset()
        now = time.time()
        
        self._last_minute_requests.append(now)
        self._daily_usage_count += 1
        self._cleanup_minute()
        self._save()

    def set_rate_limited(self, retry_after: int):
        """Set manual wait time from API response (usually minute level)"""
        self._manual_wait_until = time.time() + retry_after
        self._save()

    def set_daily_quota_exceeded(self):
        """Specifically handle 429 RESOURCE_EXHAUSTED for the day"""
        self._is_daily_quota_full = True
        # Ensure count is at least the max to show 20/20 in UI
        if self._daily_usage_count < self.max_per_day:
            self._daily_usage_count = self.max_per_day
        self._save()

    def get_seconds_until_midnight_utc(self) -> int:
        """Calculate seconds until UTC midnight (Google Quota reset)"""
        now = datetime.now(timezone.utc)
        tomorrow = now + timedelta(days=1)
        reset_time = tomorrow.replace(hour=0, minute=0, second=0, microsecond=0)
        return int((reset_time - now).total_seconds())

    def to_dict(self) -> Dict[str, Any]:
        self._check_daily_reset()
        self._cleanup_minute()
        now = time.time()
        
        used_minute = len(self._last_minute_requests)
        used_day = self._daily_usage_count
        
        # Calculate next available in seconds
        next_available_in = 0
        
        # Manual retry_after (highest priority)
        if self._manual_wait_until and self._manual_wait_until > now:
            next_available_in = int(self._manual_wait_until - now)
            
        # Per-minute limit
        elif used_minute >= self.max_per_minute:
            wait_time = int(max(1, 60 - (now - self._last_minute_requests[0])))
            next_available_in = max(next_available_in, wait_time)

        # Daily limit
        is_limit_reached = self._is_daily_quota_full or used_day >= self.max_per_day
        
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
