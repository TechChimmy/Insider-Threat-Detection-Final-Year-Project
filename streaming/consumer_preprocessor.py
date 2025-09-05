from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    'logs_raw',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest'
)

print("Listening for logs...")

for msg in consumer:
    log = json.loads(msg.value.decode())
    print(f"Received: {log}")
