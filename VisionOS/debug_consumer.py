import sys
import logging
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

from recognition.service.frame_consumer import start_consumer, _process_message
import redis
import json
import time

r = redis.from_url("redis://redis:6379", decode_responses=False)
results = r.xrevrange("visionos:frames:in", max="+", min="-", count=30)
if results:
    msg_id, fields = results[-1] # The oldest in the last 30 frames is frame 0
    print(f"Bắt đầu xử lý message cuối cùng: {msg_id}")
    print("Keys in fields:", fields.keys())
    try:
        _process_message(r, msg_id, fields)
        print("Xử lý thành công!")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("Lỗi:", e)
else:
    print("Không có message nào")
