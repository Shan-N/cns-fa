"""
api.py  —  AI IPS Real-Time Backend
=====================================
New in this version:
  • Attack type classifier  (SYN Flood / UDP Flood / ICMP Flood / HTTP Flood / Port Scan)
  • Live latency meter       (async ping loop to 127.0.0.1, sent every second)
  • Subnet tagging           (trusted 10.0.0.x vs attacker 192.168.99.x)

Start:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import json
import os
import time
import subprocess
import re
import pandas as pd
import joblib
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AI IPS - Real-Time Dashboard")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

CSV_PATH = "live_network_flows.csv"

# ── Firewall ───────────────────────────────────────────────────────
class EdgeFirewall:
    def __init__(self):
        self.blocked_ips: set = set()

    def block_ip(self, ip, reason, confidence):
        if ip not in self.blocked_ips:
            self.blocked_ips.add(ip)
            return True
        return False

    def is_blocked(self, ip):
        return ip in self.blocked_ips


# ── Attack Type Classifier ─────────────────────────────────────────
def classify_attack(raw: dict, proto_type: str) -> str:
    """
    Returns a human-readable attack label based on CIC flow features.
    Unit 3 relevance: distinguishes TCP/UDP/ICMP transport layer attacks.
    """
    try:
        proto     = str(raw.get("protocol", "")).strip()
        dst_port  = str(raw.get("dst_port", "0")).strip()
        pkts_s    = float(raw.get("flow_pkts_s", 0))
        byts_s    = float(raw.get("flow_byts_s", 0))
        fwd_pkts  = float(raw.get("tot_fwd_pkts", 0))
        bwd_pkts  = float(raw.get("tot_bwd_pkts", 0))
        duration  = float(raw.get("flow_duration", 999999))
        syn_cnt   = float(raw.get("syn_flag_cnt", 0))
        rst_cnt   = float(raw.get("rst_flag_cnt", 0))
        ack_cnt   = float(raw.get("ack_flag_cnt", 0))

        # ICMP Flood — protocol 1, no ports, high volume
        if proto in ("1", "0"):
            return "ICMP Flood"

        # UDP Flood — protocol 17, one-directional, high rate
        if proto == "17":
            if pkts_s > 1000 or (fwd_pkts > 100 and bwd_pkts == 0):
                return "UDP Flood"
            return "UDP Flood"

        # TCP-based attacks
        if proto == "6":
            # SYN Flood: high SYN, no ACK back (bwd=0), short duration
            if (syn_cnt > 0 or bwd_pkts == 0) and fwd_pkts > 50:
                if bwd_pkts == 0:
                    return "SYN Flood"

            # HTTP Flood: targeting port 80/443/8080, has responses
            if dst_port in ("80", "443", "8080") and bwd_pkts > 0 and pkts_s > 500:
                return "HTTP Flood"

            # Port Scan: many short flows, low bytes, varying ports
            if duration < 10000 and fwd_pkts <= 3 and byts_s < 1000:
                return "Port Scan"

            # Generic TCP Flood
            if pkts_s > 1000:
                return "TCP Flood"

    except (ValueError, TypeError):
        pass

    return "Unknown Attack"


# ── Subnet Classifier ──────────────────────────────────────────────
def classify_subnet(ip: str) -> dict:
    """
    Unit 2 relevance: subnetting and network layer addressing.
    Returns subnet info for the source IP.
    """
    parts = ip.split(".")
    if len(parts) != 4:
        return {"subnet": "unknown", "trust": "unknown", "cidr": ip}

    first_two = ".".join(parts[:2])
    first_oct = parts[0]

    if first_two == "10.0":
        return {"subnet": "10.0.0.0/24", "trust": "trusted",  "zone": "Internal LAN"}
    elif first_two == "192.168":
        third = parts[2]
        return {"subnet": f"192.168.{third}.0/24", "trust": "attacker", "zone": "External / Simulated Attacker"}
    elif first_oct == "172":
        return {"subnet": "172.16.0.0/12", "trust": "unknown", "zone": "Private Range B"}
    elif ip.startswith("127."):
        return {"subnet": "127.0.0.0/8",   "trust": "trusted",  "zone": "Loopback"}
    else:
        return {"subnet": f"{'.'.join(parts[:3])}.0/24", "trust": "unknown", "zone": "External"}


# ── Agent ──────────────────────────────────────────────────────────
class AI_Security_Agent:
    FEATURE_ORDER = [
        'duration', 'protocol_type', 'service', 'flag',
        'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent',
        'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
        'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds',
        'is_host_login', 'is_guest_login', 'count', 'srv_count',
        'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
        'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
        'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
        'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
        'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
        'dst_host_srv_rerror_rate',
    ]

    def __init__(self, model_path, scaler_path, encoders_path):
        self.model    = joblib.load(model_path)
        self.scaler   = joblib.load(scaler_path)
        self.encoders = joblib.load(encoders_path)
        self.firewall = EdgeFirewall()
        self.threshold = 0.85
        self.stats = {
            "total": 0, "blocked": 0, "allowed": 0,
            "protocols": {}, "services": {}, "top_attackers": {},
            "attack_types": {}
        }

    def _rule_based_score(self, raw: dict) -> float:
        score = 0.0
        try:
            pkts_s   = float(raw.get("flow_pkts_s", 0))
            byts_s   = float(raw.get("flow_byts_s", 0))
            fwd_pkts = float(raw.get("tot_fwd_pkts", 0))
            duration = float(raw.get("flow_duration", 999999))
            bwd_pkts = float(raw.get("tot_bwd_pkts", 0))
            proto    = str(raw.get("protocol", ""))

            if pkts_s > 10000:   score += 0.5
            elif pkts_s > 3000:  score += 0.35
            elif pkts_s > 1000:  score += 0.2

            if byts_s > 1_000_000: score += 0.3
            elif byts_s > 500_000: score += 0.2

            if duration < 50000 and fwd_pkts > 200:
                score += 0.25

            if fwd_pkts > 50 and bwd_pkts == 0:
                score += 0.2

            if proto in ("1", "0") and fwd_pkts > 100:
                score += 0.2

        except (ValueError, TypeError):
            pass

        return min(score, 1.0)

    def inspect(self, ip: str, kdd: dict, raw: dict = None) -> dict:
        self.stats["total"] += 1
        raw = raw or {}

        subnet_info = classify_subnet(ip)

        if self.firewall.is_blocked(ip):
            self.stats["blocked"] += 1
            return {
                "timestamp": pd.Timestamp.now().strftime("%H:%M:%S"),
                "ip": ip,
                "status": "blocked_edge",
                "action": "dropped",
                "confidence": 100.0,
                "protocol": kdd.get("protocol_type", "?"),
                "service":  kdd.get("service", "?"),
                "attack_type": "—",
                "subnet": subnet_info["subnet"],
                "trust":  subnet_info["trust"],
                "zone":   subnet_info["zone"],
            }

        # ── ML inference ──────────────────────────────────────────
        features = []
        for col in self.FEATURE_ORDER:
            val = kdd.get(col, 0.0)
            if col in ("protocol_type", "service", "flag"):
                le = self.encoders[col]
                if val not in le.classes_:
                    val = "<unknown>"
                features.append(le.transform([val])[0])
            else:
                try:
                    features.append(float(val))
                except (ValueError, TypeError):
                    features.append(0.0)

        scaled      = self.scaler.transform([features])
        is_attack   = self.model.predict(scaled)[0]
        ml_prob     = self.model.predict_proba(scaled)[0][1]

        rule_score  = self._rule_based_score(raw)
        attack_prob = max(ml_prob, rule_score)

        proto   = kdd.get("protocol_type", "unknown")
        service = kdd.get("service", "unknown")

        self.stats["protocols"][proto]  = self.stats["protocols"].get(proto, 0) + 1
        self.stats["services"][service] = self.stats["services"].get(service, 0) + 1

        result = {
            "timestamp":  pd.Timestamp.now().strftime("%H:%M:%S"),
            "ip":         ip,
            "protocol":   proto,
            "service":    service,
            "confidence": round(attack_prob * 100, 2),
            "subnet":     subnet_info["subnet"],
            "trust":      subnet_info["trust"],
            "zone":       subnet_info["zone"],
        }

        is_flagged = (is_attack == 1) or (rule_score >= 0.5)
        if is_flagged and attack_prob >= 0.5:
            attack_label = classify_attack(raw, proto)
            self.firewall.block_ip(ip, attack_label, attack_prob)
            result["status"]      = "anomaly"
            result["action"]      = "blocked"
            result["attack_type"] = attack_label
            self.stats["blocked"] += 1
            self.stats["top_attackers"][ip] = self.stats["top_attackers"].get(ip, 0) + 1
            self.stats["attack_types"][attack_label] = self.stats["attack_types"].get(attack_label, 0) + 1
        else:
            result["status"]      = "clean"
            result["action"]      = "allowed"
            result["attack_type"] = "—"
            self.stats["allowed"] += 1

        return result


# ── Load model ─────────────────────────────────────────────────────
_agent = None

def get_agent():
    global _agent
    if _agent is None:
        _agent = AI_Security_Agent("dos_rf_model.pkl", "dos_scaler.pkl", "dos_encoders.pkl")
        print("[API] Model loaded OK.")
    return _agent


# ── CIC → KDD translator ───────────────────────────────────────────
PORT_MAP = {"80": "http", "443": "https", "22": "ssh",
            "21": "ftp",  "53": "domain", "8080": "http-alt"}

def safe(row, *keys, default="0"):
    for k in keys:
        v = row.get(k)
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    return default

def map_cic_to_kdd(row: dict) -> dict:
    kdd = {col: 0.0 for col in AI_Security_Agent.FEATURE_ORDER}

    proto_num = safe(row, "protocol", "Protocol")
    if   proto_num == "6":        kdd["protocol_type"] = "tcp"
    elif proto_num == "17":       kdd["protocol_type"] = "udp"
    elif proto_num in ("0","1"):  kdd["protocol_type"] = "icmp"
    else:                         kdd["protocol_type"] = "other"

    port = safe(row, "dst_port", "Dst Port")
    kdd["service"] = PORT_MAP.get(port, "private")
    kdd["flag"]    = "SF"

    for field, cic_key in [
        ("duration",  "flow_duration"),
        ("src_bytes", "totlen_fwd_pkts"),
        ("dst_bytes", "totlen_bwd_pkts"),
        ("count",     "flow_pkts_s"),
    ]:
        try:
            v = float(safe(row, cic_key, default="0"))
            kdd[field] = v / 1_000_000 if field == "duration" else v
        except ValueError:
            kdd[field] = 0.0

    return kdd


# ── Latency pinger ─────────────────────────────────────────────────
async def measure_latency() -> float:
    """
    Ping 127.0.0.1 once and return RTT in milliseconds.
    Unit 3 relevance: demonstrates QoS degradation under attack.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "ping", "-c", "1", "-W", "1000", "127.0.0.1",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=2.0)
        output = stdout.decode()
        # macOS ping output: "round-trip min/avg/max/stddev = 0.041/0.041/0.041/0.000 ms"
        match = re.search(r"min/avg/max/\S+\s*=\s*([\d.]+)/([\d.]+)", output)
        if match:
            return float(match.group(2))  # avg RTT
    except Exception:
        pass
    return -1.0


# ── Async CSV tail ─────────────────────────────────────────────────
async def tail_csv(filepath: str):
    while not os.path.exists(filepath):
        print(f"[API] Waiting for {filepath}...")
        await asyncio.sleep(1)

    with open(filepath, "r") as f:
        header_line = f.readline()
        headers = [h.strip() for h in header_line.split(",")]
        print(f"[API] CSV ready: {len(headers)} columns")

        f.seek(0, 2)
        print(f"[API] Tailing from end of file...")

        row_count = 0
        while True:
            line = f.readline()
            if not line:
                await asyncio.sleep(0.05)
                continue
            line = line.strip()
            if not line:
                continue
            values = [v.strip() for v in line.split(",")]
            row = dict(zip(headers, values))
            if not row:
                continue
            row_count += 1
            if row_count <= 2:
                print(f"[API] Row {row_count}: src_ip={row.get('src_ip','?')} proto={row.get('protocol','?')} pkts_s={row.get('flow_pkts_s','?')}")
            yield row


# ── WebSocket: traffic stream ──────────────────────────────────────
@app.websocket("/ws/traffic")
async def stream(ws: WebSocket):
    await ws.accept()
    print("[API] Dashboard connected.")

    try:
        agent = get_agent()
    except Exception as e:
        await ws.send_text(json.dumps({"error": f"Model load failed: {e}"}))
        await ws.close()
        return

    stats_ticker = 0
    sent_count   = 0

    try:
        async for row in tail_csv(CSV_PATH):
            src_ip = safe(row, "src_ip", "Src IP", default="0.0.0.0")
            kdd    = map_cic_to_kdd(row)

            try:
                result = agent.inspect(src_ip, kdd, raw=row)
            except Exception as e:
                print(f"[API] Inference error: {e}")
                continue

            stats_ticker += 1
            if stats_ticker % 5 == 0:
                result["stats"] = {
                    "total":        agent.stats["total"],
                    "blocked":      agent.stats["blocked"],
                    "allowed":      agent.stats["allowed"],
                    "protocols":    dict(sorted(agent.stats["protocols"].items(),     key=lambda x: -x[1])[:5]),
                    "services":     dict(sorted(agent.stats["services"].items(),      key=lambda x: -x[1])[:5]),
                    "top_attackers":dict(sorted(agent.stats["top_attackers"].items(), key=lambda x: -x[1])[:5]),
                    "attack_types": dict(sorted(agent.stats["attack_types"].items(),  key=lambda x: -x[1])),
                    "blocked_ips":  list(agent.firewall.blocked_ips)[:20],
                }

            await ws.send_text(json.dumps(result))
            sent_count += 1

    except WebSocketDisconnect:
        print(f"[API] Disconnected after {sent_count} packets.")
    except Exception as e:
        import traceback
        print(f"[API] Stream error: {e}")
        traceback.print_exc()


# ── WebSocket: latency stream ──────────────────────────────────────
@app.websocket("/ws/latency")
async def latency_stream(ws: WebSocket):
    """
    Sends one ping measurement per second.
    Dashboard plots these to show QoS impact of the DDoS.
    """
    await ws.accept()
    print("[API] Latency monitor connected.")
    try:
        while True:
            rtt = await measure_latency()
            await ws.send_text(json.dumps({
                "t":   pd.Timestamp.now().strftime("%H:%M:%S"),
                "rtt": rtt
            }))
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        print("[API] Latency monitor disconnected.")
    except Exception as e:
        print(f"[API] Latency error: {e}")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _agent is not None}