from scapy.all import sniff, IP, TCP, UDP
import pandas as pd
import numpy as np
import joblib
import time
from collections import defaultdict
from plyer import notification

# =========================
# LOAD MODELS
# =========================
clf_model = joblib.load("models/rf_model.pkl")
iso_model = joblib.load("models/iso_model.pkl")
scaler = joblib.load("models/scaler.pkl")
selected_features = joblib.load("models/features.pkl")

# =========================
# FLOW STORAGE
# =========================
flows = {}
FLOW_TIMEOUT = 10   # seconds before a flow is force-flushed
MIN_PACKETS = 10    # minimum packets before prediction

def show_alert(msg):
    notification.notify(
        title="🚨 Intrusion Detection System",
        message=msg,
        timeout=5
    )

# =========================
# GET FLOW KEY
# =========================
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

# =========================
# FEATURE EXTRACTION
# =========================
def extract_features(flow, key):
    fwd_sizes = [p["size"] for p in flow["fwd"]]
    bwd_sizes = [p["size"] for p in flow["bwd"]]
    all_sizes = fwd_sizes + bwd_sizes

    if len(all_sizes) == 0:
        return None

    duration = flow["last"] - flow["start"]
    total_bytes = sum(all_sizes)
    total_packets = len(all_sizes)

    _, _, _, dport, _ = key

    # Flag counts (TCP only)
    syn_count = sum(
        1 for p in flow["fwd"] + flow["bwd"]
        if p.get("syn", False)
    )
    ack_count = sum(
        1 for p in flow["fwd"] + flow["bwd"]
        if p.get("ack", False)
    )

    fwd_sizes_arr = np.array(fwd_sizes) if fwd_sizes else np.array([0])
    bwd_sizes_arr = np.array(bwd_sizes) if bwd_sizes else np.array([0])
    all_sizes_arr = np.array(all_sizes)

    features = {
        "Total Fwd Packets":            len(fwd_sizes),
        "Total Backward Packets":       len(bwd_sizes),
        "Total Length of Fwd Packets":  float(np.sum(fwd_sizes_arr)),
        "Total Length of Bwd Packets":  float(np.sum(bwd_sizes_arr)),
        "Flow Duration":                duration,
        "Flow Bytes/s":                 total_bytes / duration if duration > 0 else 0,
        "Flow Packets/s":               total_packets / duration if duration > 0 else 0,
        "Packet Length Mean":           float(np.mean(all_sizes_arr)),
        "Packet Length Std":            float(np.std(all_sizes_arr)),
        "Average Packet Size":          float(np.mean(all_sizes_arr)),
        "Min Packet Length":            float(np.min(all_sizes_arr)),
        "Max Packet Length":            float(np.max(all_sizes_arr)),
        "Fwd Packet Length Mean":       float(np.mean(fwd_sizes_arr)),
        "Bwd Packet Length Mean":       float(np.mean(bwd_sizes_arr)),
        "Fwd Packets/s":                len(fwd_sizes) / duration if duration > 0 else 0,
        "Bwd Packets/s":                len(bwd_sizes) / duration if duration > 0 else 0,
        "SYN Flag Count":               syn_count,
        "ACK Flag Count":               ack_count,
        "Destination Port":             dport,
    }

    return features

# =========================
# MODEL PREDICTION
# =========================
def run_model(features, key):
    df = pd.DataFrame([features])

    # fill any missing features with 0
    for col in selected_features:
        if col not in df.columns:
            df[col] = 0

    df = df[selected_features]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    df_scaled = scaler.transform(df)

    clf_proba  = clf_model.predict_proba(df_scaled)[0][1]
    iso_score  = iso_model.decision_function(df_scaled)[0]

    src, dst, sport, dport, _ = key
    flow_info = f"{src}:{sport} → {dst}:{dport}"

    if clf_proba > 0.5:
        label = "🚨 KNOWN ATTACK"
    elif iso_score < -0.3:
        label = "⚠️  UNKNOWN ATTACK"
    else:
        label = "✅ NORMAL"

    if "ATTACK" in label:
        show_alert(label)

    print(
        f"[{time.strftime('%H:%M:%S')}] {flow_info:<45} "
        f"{label:<20} "
        f"clf={clf_proba:.2f}  iso={iso_score:.3f}"
    )

    df_out = pd.DataFrame([features])

    df_out["Label"] = label
    df_out["CLF_Prob"] = clf_proba
    df_out["ISO_Score"] = iso_score
    df_out["Timestamp"] = time.strftime('%H:%M:%S')

    # append to csv
    df_out.to_csv("live_data.csv", mode="a", header=not pd.io.common.file_exists("live_data.csv"), index=False)

# =========================
# PROCESS PACKET
# =========================
def process_packet(packet):
    forward_key, reverse_key = get_flow_key(packet)
    if forward_key is None:
        return

    now  = time.time()
    size = len(packet)

    # extract TCP flags safely
    syn = ack = False
    if packet.haslayer(TCP):
        flags = packet[TCP].flags
        syn   = bool(flags & 0x02)
        ack   = bool(flags & 0x10)

    pkt_info = {"size": size, "time": now, "syn": syn, "ack": ack}

    # find which direction this packet belongs to
    if forward_key in flows:
        active_key = forward_key
        direction  = "fwd"
    elif reverse_key in flows:
        active_key = reverse_key
        direction  = "bwd"
    else:
        # new flow
        flows[forward_key] = {
            "start": now,
            "last":  now,
            "fwd":   [pkt_info],
            "bwd":   []
        }
        return

    flow = flows[active_key]
    flow[direction].append(pkt_info)
    flow["last"] = now                  # ← bug fix: was missing before

    total_packets = len(flow["fwd"]) + len(flow["bwd"])
    elapsed       = now - flow["start"]

    should_flush = (
        total_packets >= MIN_PACKETS or
        elapsed > FLOW_TIMEOUT
    )

    if should_flush:
        features = extract_features(flow, active_key)
        if features:
            run_model(features, active_key)
        del flows[active_key]

# =========================
# PERIODIC TIMEOUT FLUSH
# (cleans up flows that received no packets recently)
# =========================
def flush_stale_flows():
    now = time.time()
    stale = [k for k, v in flows.items() if now - v["last"] > FLOW_TIMEOUT]
    for k in stale:
        features = extract_features(flows[k], k)
        if features:
            run_model(features, k)
        del flows[k]

# =========================
# START
# =========================
print("🚀 IDS Running — press Ctrl+C to stop\n")
print(f"{'Timestamp':<12} {'Flow':<45} {'Result':<20} {'Scores'}")
print("-" * 90)

try:
    while True:
        sniff(prn=process_packet, store=False, timeout=5)
        flush_stale_flows()
except KeyboardInterrupt:
    print("\n IDS stopped.")