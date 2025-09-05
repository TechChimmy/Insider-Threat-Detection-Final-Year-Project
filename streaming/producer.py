import os
import json
from datetime import datetime
from kafka import KafkaConsumer, KafkaProducer

# Ensure processed data directory exists
os.makedirs("data/processed", exist_ok=True)

# Kafka setup
RAW_TOPIC = "logs_raw"
PREPROCESSED_TOPIC = "logs_preprocessed"
BOOTSTRAP_SERVERS = "localhost:9092"

consumer = KafkaConsumer(
    RAW_TOPIC,
    bootstrap_servers=BOOTSTRAP_SERVERS,
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    group_id="spynet-preprocessor",
    value_deserializer=lambda m: json.loads(m.decode("utf-8"))
)

producer = KafkaProducer(
    bootstrap_servers=BOOTSTRAP_SERVERS,
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

# File for offline storage
output_file = "data/processed/processed_logs.jsonl"

def preprocess(log):
    """
    Simple preprocessing:
    - Normalize timestamp
    - Map event types to IDs
    - Extract hour of day (behavioral pattern)
    """
    ts = datetime.fromtimestamp(log.get("timestamp", datetime.now().timestamp()))
    
    event_map = {"login": 0, "file_access": 1, "email": 2, "http": 3, "device": 4}
    event_type = log.get("event", "unknown")
    event_id = event_map.get(event_type, -1)

    processed = {
        "user": log.get("user", "unknown"),
        "event": event_type,
        "event_id": event_id,
        "timestamp": ts.isoformat(),
        "hour": ts.hour,
        "raw_ts": log.get("timestamp", None)
    }
    return processed

print("📡 SpyNet Preprocessor started... Listening for logs.")

with open(output_file, "a") as f:
    for msg in consumer:
        raw_log = msg.value
        processed_log = preprocess(raw_log)

        # Send to Kafka preprocessed topic
        producer.send(PREPROCESSED_TOPIC, processed_log)

        # Save offline
        f.write(json.dumps(processed_log) + "\n")

        print(f"✅ Processed log -> {processed_log}")
