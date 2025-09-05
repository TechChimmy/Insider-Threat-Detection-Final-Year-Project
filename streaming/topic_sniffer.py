from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    "logs_preprocessed",
    bootstrap_servers="localhost:9092",
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    group_id="sniffer",
    value_deserializer=lambda m: json.loads(m.decode("utf-8"))
)

print("🔎 Listening on logs_preprocessed ...\n")
for msg in consumer:
    print("📥", msg.value)
