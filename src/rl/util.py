import time
import functools

def profile_time(name=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tag = name or func.__name__
            start = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start
            print(f"[PROFILE] {tag} took {duration:.4f} sec")
            return result
        return wrapper
    return decorator

# 데이터셋에 대한 적합성 여부 검토
# 정의된 환경에 대한 적합성 여부 검토
# 학습 과정 중 적합성 여부 검토