import time
import random
import numpy as np
import pandas as pd
import joblib
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

warnings.filterwarnings('ignore')

# ==========================================
# Phase 1: Model Training & Serialization
# ==========================================
def train_and_save_model(train_csv_path):
    print("[SYSTEM] Loading Real Data & Training AI Model...")
    df = pd.read_csv(train_csv_path)
    
    # 1. Separate Features and Target
    X = df.drop(columns=['class'])
    
    # Encode target: normal = 0, anomaly = 1
    y = df['class'].apply(lambda x: 0 if x == 'normal' else 1).values
    
    # 2. Encode Categorical Features
    cat_cols = ['protocol_type', 'service', 'flag']
    encoders = {}
    
    for col in cat_cols:
        le = LabelEncoder()
        # Adding an '<unknown>' class to handle any unseen data in the test set gracefully
        unique_vals = list(X[col].unique()) + ['<unknown>']
        le.fit(unique_vals)
        X[col] = le.transform(X[col])
        encoders[col] = le
        
    # 3. Scale Features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 4. Train Model
    print("[SYSTEM] Training Random Forest on 25,000+ records. Please wait...")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf_model.fit(X_scaled, y)
    
    # 5. Save Artifacts for Production
    joblib.dump(rf_model, 'dos_rf_model.pkl')
    joblib.dump(scaler, 'dos_scaler.pkl')
    joblib.dump(encoders, 'dos_encoders.pkl')
    print("[SYSTEM] Model, Scaler, and Encoders saved to disk successfully.\n")

# ==========================================
# Phase 2: Automated Intrusion Prevention
# ==========================================
class EdgeFirewall:
    def __init__(self):
        self.blocked_ips = set()

    def block_ip(self, ip_address, reason, confidence):
        if ip_address not in self.blocked_ips:
            self.blocked_ips.add(ip_address)
            # Example system call: os.system(f"iptables -A INPUT -s {ip_address} -j DROP")
            print(f"[FIREWALL] 🚨 ACTION TAKEN: Dropping traffic from {ip_address}")
            print(f"           Reason: {reason} (AI Confidence: {confidence:.2%})")
            
    def is_blocked(self, ip_address):
        return ip_address in self.blocked_ips

class AI_Security_Agent:
    def __init__(self, model_path, scaler_path, encoders_path):
        # Load the trained model, scaler, and encoders into memory
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.encoders = joblib.load(encoders_path)
        self.firewall = EdgeFirewall()
        self.threshold = 0.85 # 85% confidence required to automate a block
        self.cat_cols = ['protocol_type', 'service', 'flag']

    def inspect_traffic(self, ip_address, raw_feature_dict):
        """Processes a single network connection dictionary through the ML pipeline."""
        if self.firewall.is_blocked(ip_address):
            # Drop silently if already blocked to save compute at the edge
            return 
            
        # 1. Format and Encode incoming raw data
        features = []
        for col in raw_feature_dict.keys():
            val = raw_feature_dict[col]
            if col in self.cat_cols:
                le = self.encoders[col]
                # Fallback to <unknown> if the test data has a completely new protocol/service
                if val not in le.classes_:
                    val = '<unknown>'
                features.append(le.transform([val])[0])
            else:
                features.append(float(val))
                
        # 2. Scale
        scaled_features = self.scaler.transform([features])
        
        # 3. AI Inference
        is_attack = self.model.predict(scaled_features)[0]
        attack_prob = self.model.predict_proba(scaled_features)[0][1]
        
        # 4. Execution Logic
        if is_attack == 1:
            print(f"[MONITOR] ⚠️  Anomaly detected from {ip_address} | Protocol: {raw_feature_dict['protocol_type']}, Service: {raw_feature_dict['service']}")
            if attack_prob >= self.threshold:
                self.firewall.block_ip(ip_address, "ML_Anomaly_Signature", attack_prob)
            else:
                print(f"[MONITOR] 🔍 Suspicious ({attack_prob:.2%}). Confidence too low for auto-block. Logging.")
        else:
            print(f"[MONITOR] ✅ Traffic from {ip_address} looks clean.")

# ==========================================
# Phase 3: Execution Loop
# ==========================================
if __name__ == "__main__":
    # 1. Ensure the model is trained on your Train_data.csv
    train_and_save_model('Train_data.csv')
    
    # 2. Spin up the security agent
    print("Starting Live Network Monitoring...\n")
    agent = AI_Security_Agent('dos_rf_model.pkl', 'dos_scaler.pkl', 'dos_encoders.pkl')
    
    # 3. Simulate Live Traffic Stream using your Test_data.csv
    print("[SYSTEM] Loading unseen test data to simulate live network flow...")
    test_df = pd.read_csv('Test_data.csv')
    
    # Mock pool of IPs hitting our server
    mock_ips = [f"192.168.1.{i}" for i in range(10, 30)]
    
    # Randomly sample 10 connections from the Test dataset to simulate live traffic
    sample_traffic = test_df.sample(10, random_state=42).to_dict(orient='records')
    
    for packet_data in sample_traffic:
        time.sleep(1) # Simulate network arrival time
        
        # Assign a random IP to the packet
        source_ip = random.choice(mock_ips)
        
        # Process the raw dictionary through our IPS agent
        agent.inspect_traffic(source_ip, packet_data)
        print("-" * 75)