"""Run the pinned veRL trainer and release its local Ray cluster on exit."""
import ray
from verl.experimental.fully_async_policy.fully_async_main import main

if __name__ == '__main__':
    try:
        main()
    finally:
        ray.shutdown()
