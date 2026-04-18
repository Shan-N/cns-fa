"""
ddos_sim.py  —  Local DDoS Simulator (loopback only, safe self-test)
=====================================================================
Floods 127.0.0.1 with SYN / UDP / ICMP packets using Scapy, then
writes synthetic CICFlowMeter-style rows to live_network_flows.csv
so api.py can pick them up in real-time.

Usage (requires root for raw sockets):
    sudo python3 ddos_sim.py

Dependencies:
    pip install scapy
"""

import csv
import os
import random
import signal
import sys
import time
import threading
from datetime import datetime

try:
    from scapy.all import IP, TCP, UDP, ICMP, send, conf
    conf.verb = 0  # suppress scapy output
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("[WARNING] Scapy not installed. Running in CSV-only simulation mode.")
    print("          Install with: pip install scapy")

# ── Configuration ─────────────────────────────────────────────────
TARGET_IP       = "127.0.0.1"   # loopback ONLY – safe self-test
ATTACK_PORTS    = [80, 443, 8080, 22, 53]
NORMAL_IPS      = [f"10.0.0.{i}"    for i in range(1, 20)]
ATTACKER_IPS    = [f"192.168.99.{i}" for i in range(1, 15)]
OUTPUT_CSV      = "live_network_flows.csv"
PACKET_INTERVAL = 0.08          # seconds between CSV rows (≈12 rows/s)
BURST_SIZE      = 5             # packets per burst in "attack" mode

# ── CICFlowMeter CSV columns (must match api.py expectations) ──────
CSV_COLUMNS = [
    "src_ip","dst_ip","src_port","dst_port","Protocol","timestamp",
    "Flow Duration","flow_byts_s","Flow Pkts/s",
    "fwd_pkts_s","bwd_pkts_s","tot_fwd_pkts","tot_bwd_pkts",
    "TotLen Fwd Pkts","TotLen Bwd Pkts",
    "fwd_pkt_len_max","fwd_pkt_len_min","fwd_pkt_len_mean","fwd_pkt_len_std",
    "bwd_pkt_len_max","bwd_pkt_len_min","bwd_pkt_len_mean","bwd_pkt_len_std",
    "pkt_len_max","pkt_len_min","pkt_len_mean","pkt_len_std","pkt_len_var",
    "fwd_header_len","bwd_header_len",
    "Src IP",   # extra alias column – api.py reads this
]

# ── Scapy packet flooding (raw sockets) ────────────────────────────

def syn_flood(target_ip, port, count=BURST_SIZE):
    if not SCAPY_AVAILABLE:
        return
    for _ in range(count):
        src_port = random.randint(1024, 65535)
        pkt = IP(dst=target_ip) / TCP(sport=src_port, dport=port, flags="S")
        send(pkt, iface="lo0", verbose=False)

def udp_flood(target_ip, port, count=BURST_SIZE):
    if not SCAPY_AVAILABLE:
        return
    for _ in range(count):
        payload = b"X" * random.randint(50, 512)
        pkt = IP(dst=target_ip) / UDP(sport=random.randint(1024, 65535), dport=port) / payload
        send(pkt, iface="lo0", verbose=False)

def icmp_flood(target_ip, count=BURST_SIZE):
    if not SCAPY_AVAILABLE:
        return
    for _ in range(count):
        pkt = IP(dst=target_ip) / ICMP()
        send(pkt, iface="lo0", verbose=False)

# ── CSV row generators ──────────────────────────────────────────────

def make_attack_row(src_ip, proto, port):
    """High-rate, high-volume row that should trigger the ML model."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    duration = random.uniform(0.001, 0.05)           # very short flows = DoS
    fwd_bytes = random.randint(5000, 50000)
    pkt_rate  = random.uniform(20,50)          # packets/s = extreme
    return {
        "src_ip": src_ip,
        "dst_ip": TARGET_IP,
        "src_port": random.randint(1024, 65535),
        "dst_port": port,
        "Protocol": "6" if proto == "tcp" else ("17" if proto == "udp" else "1"),
        "timestamp": now,
        "Flow Duration": int(duration * 1_000_000),
        "flow_byts_s": fwd_bytes / max(duration, 0.001),
        "Flow Pkts/s": pkt_rate,
        "fwd_pkts_s": pkt_rate,
        "bwd_pkts_s": 0,
        "tot_fwd_pkts": random.randint(500, 5000),
        "tot_bwd_pkts": 0,
        "TotLen Fwd Pkts": fwd_bytes,
        "TotLen Bwd Pkts": 0,
        "fwd_pkt_len_max": 1500,
        "fwd_pkt_len_min": 64,
        "fwd_pkt_len_mean": random.uniform(500, 1400),
        "fwd_pkt_len_std": random.uniform(200, 600),
        "bwd_pkt_len_max": 0, "bwd_pkt_len_min": 0,
        "bwd_pkt_len_mean": 0, "bwd_pkt_len_std": 0,
        "pkt_len_max": 1500, "pkt_len_min": 64,
        "pkt_len_mean": random.uniform(500, 1400),
        "pkt_len_std": random.uniform(200, 600),
        "pkt_len_var": random.uniform(40000, 360000),
        "fwd_header_len": 20, "bwd_header_len": 0,
        "Src IP": src_ip,
    }

def make_normal_row(src_ip, port):
    """Low-rate, low-volume row that should be allowed."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    duration = random.uniform(0.5, 5.0)
    fwd_bytes = random.randint(200, 2000)
    pkt_rate  = random.uniform(1, 50)
    return {
        "src_ip": src_ip,
        "dst_ip": TARGET_IP,
        "src_port": random.randint(1024, 65535),
        "dst_port": port,
        "Protocol": random.choice(["6", "17"]),
        "timestamp": now,
        "Flow Duration": int(duration * 1_000_000),
        "flow_byts_s": fwd_bytes / duration,
        "Flow Pkts/s": pkt_rate,
        "fwd_pkts_s": pkt_rate * 0.6,
        "bwd_pkts_s": pkt_rate * 0.4,
        "tot_fwd_pkts": random.randint(3, 30),
        "tot_bwd_pkts": random.randint(2, 20),
        "TotLen Fwd Pkts": fwd_bytes,
        "TotLen Bwd Pkts": random.randint(100, 1000),
        "fwd_pkt_len_max": 512,
        "fwd_pkt_len_min": 40,
        "fwd_pkt_len_mean": random.uniform(80, 400),
        "fwd_pkt_len_std": random.uniform(20, 100),
        "bwd_pkt_len_max": 256, "bwd_pkt_len_min": 40,
        "bwd_pkt_len_mean": random.uniform(60, 200),
        "bwd_pkt_len_std": random.uniform(10, 60),
        "pkt_len_max": 512, "pkt_len_min": 40,
        "pkt_len_mean": random.uniform(80, 300),
        "pkt_len_std": random.uniform(20, 80),
        "pkt_len_var": random.uniform(400, 6400),
        "fwd_header_len": 20, "bwd_header_len": 20,
        "Src IP": src_ip,
    }

# ── Main simulation loop ────────────────────────────────────────────

stop_flag = threading.Event()

def signal_handler(sig, frame):
    print("\n[SIM] Stopping simulation...")
    stop_flag.set()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def run_simulation():
    # Create/overwrite CSV with headers
    file_exists = os.path.exists(OUTPUT_CSV)
    
    with open(OUTPUT_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        
        print(f"[SIM] Writing to {OUTPUT_CSV}  |  Attack IPs: {len(ATTACKER_IPS)}  |  Normal IPs: {len(NORMAL_IPS)}")
        print(f"[SIM] Press Ctrl+C to stop\n")
        
        phase_counter = 0
        
        while not stop_flag.is_set():
            phase_counter += 1
            
            # ── Alternating phases: 10s normal → 10s DDoS ──
            in_attack_phase = (phase_counter % 20) >= 10
            
            if in_attack_phase:
                # DDoS burst: many rows from attacker IPs
                proto = random.choice(["tcp", "udp", "icmp"])
                port  = random.choice(ATTACK_PORTS)
                src   = random.choice(ATTACKER_IPS)
                
                print(f"[SIM] 🔴 ATTACK  | src={src} proto={proto} port={port}")
                
                # Write multiple rows fast (simulates burst)
                for _ in range(BURST_SIZE):
                    if proto == "icmp":
                        row = make_attack_row(src, "icmp", 0)
                    else:
                        row = make_attack_row(src, proto, port)
                    writer.writerow(row)
                    f.flush()
                
                # Optional raw packet flood (requires sudo + scapy)
                if SCAPY_AVAILABLE:
                    t = threading.Thread(
                        target=(syn_flood if proto == "tcp" else
                                udp_flood if proto == "udp" else
                                lambda ip, **_: icmp_flood(ip)),
                        args=(TARGET_IP, port) if proto != "icmp" else (TARGET_IP,),
                        daemon=True
                    )
                    t.start()
            else:
                # Normal traffic: single row from a normal IP
                src  = random.choice(NORMAL_IPS)
                port = random.choice(ATTACK_PORTS)
                print(f"[SIM] 🟢 NORMAL  | src={src} port={port}")
                row = make_normal_row(src, port)
                writer.writerow(row)
                f.flush()
            
            time.sleep(PACKET_INTERVAL)

if __name__ == "__main__":
    run_simulation()