import os
import redis


def local_redis_set(name):
    redis_pool = redis.ConnectionPool(host='127.0.0.1', port=6379, db=0)
    redis_connection = redis.Redis(connection_pool=redis_pool, decode_responses=True)

    pid = os.getpid()
    redis_connection.hset(name, 'pid', pid)
