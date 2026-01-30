from src.services.rate_tracker import rate_tracker

class RateLimitController:
    @staticmethod
    async def get_status():
        """
        Get current rate limit status
        """
        return await rate_tracker.to_dict()
