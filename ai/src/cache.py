import redis
import json

class Cache:
    def __init__(self, host="localhost", port=6379):
        self.client = redis.Redis(host=host, port=port, decode_responses=True)
    
    def get(self, key):
        result = self.client.get(key)
        return json.loads(result) if result else None
    
    def set(self, key, value, expire=3600):
        self.client.setex(key, expire, json.dumps(value))