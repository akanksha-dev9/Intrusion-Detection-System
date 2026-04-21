from scapy.all import sniff, IP, TCP, UDP
import pandas as pd
import numpy as np
import joblib
import time
from collections import defaultdict
from plyer import notification

# Load models
clf_model = joblib.load("models/xgb_model.pkl")
iso_model = joblib.load("models/iso_model.pkl")
selected_features = joblib.load("models/features.pkl")

# flow storage
flows = {}
FLOW_TIMEOUT = 10   # seconds before a flow is force-flushed
MIN_PACKETS = 10    # minimum packets before prediction

def show_alert(msg):
    notification.notify(
        title="🚨 Attack detected",
        message=msg,
        timeout=5
    )


# Get flow key
def get_flow_key(packet):
    if not packet.haslayer(IP):
        return None, None

    src = packet[IP].src
    dst = packet[IP].dst
    proto = packet[IP].proto
    sport, dport = 0, 0

    if packet.haslayer(TCP):
        sport = packet[TCP].sport
        dport = packet[TCP].dport
    elif packet.haslayer(UDP):
        sport = packet[UDP].sport
        dport = packet[UDP].dport

    forward_key = (src, dst, sport, dport, proto)
    reverse_key  = (dst, src, dport, sport, proto)

    return forward_key, reverse_key


# Feature extraction
def extract_features(flow, key):
    fwd_sizes = [p["size"] for p in flow["fwd"]]
    bwd_sizes = [p["size"] for p in flow["bwd"]]
    all_sizes = fwd_sizes + bwd_sizes

    if len(all_sizes) == 0:
        return None

    duration = flow["last"] - flow["start"]
    total_bytes = sum(all_sizes)

    _, _, _, dport, _ = key

    # convert to arrays safely
    fwd_sizes_arr = np.array(fwd_sizes) if fwd_sizes else np.array([0])
    bwd_sizes_arr = np.array(bwd_sizes) if bwd_sizes else np.array([0])
    all_sizes_arr = np.array(all_sizes)

    features = {
        "Average Packet Size": float(np.mean(all_sizes_arr)),
        "Packet Length Std": float(np.std(all_sizes_arr)),
        "Bwd Packet Length Mean": float(np.mean(bwd_sizes_arr)),
        "Packet Length Mean": float(np.mean(all_sizes_arr)),
        "Max Packet Length": float(np.max(all_sizes_arr)),
        "Destination Port": dport,
        "Total Length of Fwd Packets": float(np.sum(fwd_sizes_arr)),
        "Total Length of Bwd Packets": float(np.sum(bwd_sizes_arr)),
        "Fwd Packet Length Mean": float(np.mean(fwd_sizes_arr)),
        "Total Fwd Packets": len(fwd_sizes),
        "Min Packet Length": float(np.min(all_sizes_arr)),
        "Flow Bytes/s": total_bytes / duration if duration > 0 else 0,
    }

    return features

# Hybrid Model prediction
def run_model(features, key):
    df = pd.DataFrame([features])

    # ensure feature consistency
    for col in selected_features:
        if col not in df.columns:
            df[col] = 0

    df = df[selected_features]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    sample = df

    # model outputs
    clf_proba = clf_model.predict_proba(sample)[0][1]
    iso_score = iso_model.decision_function(sample)[0]

    STREAMING_PORTS = {443, 80, 8080}  # HTTPS, HTTP
    _, _, _, dport, _ = key

    clf_threshold = 1.0 if dport in STREAMING_PORTS else 0.5

    # decision logic
    if clf_proba > clf_threshold:
        label = "KNOWN ATTACK"

    elif clf_proba > 0.4 and iso_score < -0.2:
        label = "UNKNOWN ATTACK"

    elif iso_score < -0.25:
        label = "SUSPICIOUS"

    else:
        label = "NORMAL"

    # alert
    if "ATTACK" in label or "SUSPICIOUS" in label:
        show_alert(label)

    src, dst, sport, dport, _ = key
    flow_info = f"{src}:{sport} → {dst}:{dport}"

    print(
        f"[{time.strftime('%H:%M:%S')}] {flow_info:<45} "
        f"{label:<20} clf={clf_proba:.2f} iso={iso_score:.3f}"
    )

    # save output
    df_out = pd.DataFrame([features])
    df_out["Label"] = label
    df_out["CLF_Prob"] = clf_proba
    df_out["ISO_Score"] = iso_score
    df_out["Timestamp"] = time.strftime('%H:%M:%S')

    df_out.to_csv("live_data.csv", mode="a",
                  header=not pd.io.common.file_exists("live_data.csv"),
                  index=False)

# Process packet
def process_packet(packet):
    forward_key, reverse_key = get_flow_key(packet)
    if forward_key is None:
        return

    now = time.time()
    size = len(packet)

    syn = ack = False
    if packet.haslayer(TCP):
        flags = packet[TCP].flags
        syn = bool(flags & 0x02)
        ack = bool(flags & 0x10)

    pkt_info = {"size": size, "time": now, "syn": syn, "ack": ack}

    if forward_key in flows:
        active_key = forward_key
        direction = "fwd"
    elif reverse_key in flows:
        active_key = reverse_key
        direction = "bwd"
    else:
        flows[forward_key] = {
            "start": now,
            "last": now,
            "fwd": [pkt_info],
            "bwd": []
        }
        return

    flow = flows[active_key]
    flow[direction].append(pkt_info)
    flow["last"] = now

    total_packets = len(flow["fwd"]) + len(flow["bwd"])
    elapsed = now - flow["start"]

    if total_packets >= MIN_PACKETS or elapsed > FLOW_TIMEOUT:
        features = extract_features(flow, active_key)
        if features:
            run_model(features, active_key)
        del flows[active_key]

# Periodic timeout flush
# (cleans up flows that received no packets recently)
def flush_stale_flows():
    now = time.time()
    stale = [k for k, v in flows.items() if now - v["last"] > FLOW_TIMEOUT]
    for k in stale:
        features = extract_features(flows[k], k)
        if features:
            run_model(features, k)
        del flows[k]

# Start
print("🚀 IDS Running — press Ctrl+C to stop\n")
print(f"{'Timestamp':<12} {'Flow':<45} {'Result':<20} {'Scores'}")
print("-" * 90)

try:
    while True:
        sniff(prn=process_packet, store=False, timeout=5)
        flush_stale_flows()
except KeyboardInterrupt:
    print("\n IDS stopped.")