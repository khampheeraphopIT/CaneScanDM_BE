"""
Rate Limit Tracker - ติดตามการใช้งาน API
"""
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

@dataclass
class RateLimitInfo:
    requests_per_minute: int  # จำนวนครั้งที่ใช้ใน 1 นาที
    requests_per_day: int     # จำนวนครั้งที่ใช้ใน 1 วัน
    max_per_minute: int       # จำกัดต่อนาที
    max_per_day: int          # จำกัดต่อวัน
    next_available_in: int    # วินาทีจนกว่าจะใช้ได้
    can_request: bool         # สามารถใช้ได้หรือไม่

class RateLimitTracker:
    """Track API usage locally since Gemini doesn't provide quota API"""
    
    def __init__(self, max_per_minute: int = 2, max_per_day: int = 50):
        self.max_per_minute = max_per_minute
        self.max_per_day = max_per_day
        self.requests: deque = deque()  # Store timestamps
        self._last_rate_limit_until: Optional[float] = None
    
    def _cleanup_old_requests(self):
        """Remove requests older than 24 hours"""
        now = time.time()
        day_ago = now - 86400  # 24 hours
        while self.requests and self.requests[0] < day_ago:
            self.requests.popleft()
    
    def record_request(self):
        """Record a new request"""
        self.requests.append(time.time())
        self._cleanup_old_requests()
    
    def set_rate_limited(self, retry_after: int):
        """Set rate limit from API response"""
        self._last_rate_limit_until = time.time() + retry_after
    
    def get_info(self) -> RateLimitInfo:
        """Get current rate limit info"""
        now = time.time()
        self._cleanup_old_requests()
        
        # Count requests in last minute
        minute_ago = now - 60
        requests_in_minute = sum(1 for t in self.requests if t > minute_ago)
        
        # Count requests in last day
        requests_in_day = len(self.requests)
        
        # Check if rate limited
        next_available_in = 0
        if self._last_rate_limit_until and now < self._last_rate_limit_until:
            next_available_in = int(self._last_rate_limit_until - now)
        elif requests_in_minute >= self.max_per_minute:
            # Calculate when oldest request in minute window expires
            oldest_in_minute = min((t for t in self.requests if t > minute_ago), default=now)
            next_available_in = max(0, int(60 - (now - oldest_in_minute)))
        
        can_request = (
            requests_in_minute < self.max_per_minute and 
            requests_in_day < self.max_per_day and
            next_available_in == 0
        )
        
        return RateLimitInfo(
            requests_per_minute=requests_in_minute,
            requests_per_day=requests_in_day,
            max_per_minute=self.max_per_minute,
            max_per_day=self.max_per_day,
            next_available_in=next_available_in,
            can_request=can_request
        )
    
    def to_dict(self) -> dict:
        """Convert to dict for API response"""
        info = self.get_info()
        return {
            "requests_used_minute": info.requests_per_minute,
            "requests_used_day": info.requests_per_day,
            "max_per_minute": info.max_per_minute,
            "max_per_day": info.max_per_day,
            "remaining_minute": max(0, info.max_per_minute - info.requests_per_minute),
            "remaining_day": max(0, info.max_per_day - info.requests_per_day),
            "next_available_in": info.next_available_in,
            "can_request": info.can_request
        }

# Global instance
rate_tracker = RateLimitTracker(max_per_minute=2, max_per_day=50)
