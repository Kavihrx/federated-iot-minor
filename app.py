from flask import Flask, jsonify, request, render_template
import numpy as np
import random
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# --- Configuration ---
NUM_CLIENTS = 5
NUM_FEATURES = 10
ANOMALY_THRESHOLD = 0.70

# --- Simulated Client Models ---
# Using Logistic Regression to simulate local ML models on IoT devices
client_models = [LogisticRegression() for _ in range(NUM_CLIENTS)]

def generate_dummy_data(is_anomalous=False):
    """Simulate IoT network traffic data (e.g., packet size, inter-arrival time)."""
    # We must include both classes (0 and 1) so Logistic Regression can fit properly.
    if is_anomalous:
        # Mostly anomalous data
        X_normal = np.random.randn(10, NUM_FEATURES)
        y_normal = np.zeros(10)
        X_anom = np.random.randn(40, NUM_FEATURES) * 2 + 1.5
        y_anom = np.ones(40)
    else:
        # Mostly normal data
        X_normal = np.random.randn(45, NUM_FEATURES)
        y_normal = np.zeros(45)
        X_anom = np.random.randn(5, NUM_FEATURES) * 2 + 1.5
        y_anom = np.ones(5)
        
    X = np.vstack((X_normal, X_anom))
    y = np.hstack((y_normal, y_anom))
    return X, y

def simulate_llm_classification(anomaly_score):
    """
    Simulates a lightweight LLM-inspired classification module.
    In the paper, an LLM would process the textual anomaly logs.
    Here, we use a heuristic to classify the attack type and recommend mitigations.
    """
    if anomaly_score < 0.5:
        return {
            "status": "Normal",
            "attack_type": "None",
            "recommendation": "Maintain standard logging."
        }
    elif 0.5 <= anomaly_score < 0.8:
        return {
            "status": "Warning",
            "attack_type": "Reconnaissance Port Scan",
            "recommendation": "Block scanner IP. Isolate IoT device on VLAN."
        }
    else:
        return {
            "status": "CRITICAL",
            "attack_type": "DDoS Payload Flood",
            "recommendation": "Null-route source IP. Trigger automated edge-firewall block."
        }

@app.route('/api/federated_pipeline', methods=['GET', 'POST'])
def run_pipeline():
    client_updates_weights = []
    client_updates_bias = []
    client_anomaly_logs = []
    
    # --- PHASE 1: Client Local Training & Anomaly Detection ---
    all_scores = []
    for i in range(NUM_CLIENTS):
        # Randomly decide if this client experiences anomalous traffic (~40% chance for visual demo)
        experiences_anomaly = random.random() > 0.60
        X, y = generate_dummy_data(is_anomalous=experiences_anomaly)
        
        # Local Training Simulator
        model = client_models[i]
        
        # Train locally
        model.fit(X, y)
        
        # Store weights for federated averaging
        client_updates_weights.append(model.coef_[0])
        client_updates_bias.append(model.intercept_)
        
        # Local Anomaly Detection (Simulated by checking mean prediction probability of class 1)
        preds = model.predict_proba(X)[:, 1]
        mean_anomaly_score = np.mean(preds)
        
        
        all_scores.append(round(float(mean_anomaly_score), 4))
        
        if experiences_anomaly or mean_anomaly_score > ANOMALY_THRESHOLD:
            client_anomaly_logs.append({
                "client_id": f"IoT_Client_{i+1}",
                "anomaly_score": round(float(mean_anomaly_score), 4)
            })

    # --- PHASE 2: Server Aggregation (Federated Averaging) ---
    # Aggregate weights to form the global model update
    global_weights = np.mean(client_updates_weights, axis=0)
    global_bias = np.mean(client_updates_bias, axis=0)
    
    # --- PHASE 3: LLM-Driven Classification ---
    # Process the aggregated anomaly logs using the simulated LLM logic
    final_alerts = []
    if not client_anomaly_logs:
        result = simulate_llm_classification(0.1)
        final_alerts.append({
            "status": result["status"],
            "attack_type": result["attack_type"],
            "recommendation": result["recommendation"],
            "details": "No anomalies detected across all clients."
        })
    else:
        for log in client_anomaly_logs:
            classification = simulate_llm_classification(log['anomaly_score'])
            final_alerts.append({
                "client_id": log['client_id'],
                "status": classification["status"],
                "attack_type": classification["attack_type"],
                "recommendation": classification["recommendation"],
                "anomaly_score": log['anomaly_score']
            })
            
    # Return aggregated result with the classification outputs
    response = {
        "message": "Federated rounds completed successfully.",
        "federated_server_status": "Weights Aggregated (FedAvg)",
        "results": final_alerts,
        "all_scores": all_scores
    }
    
    return jsonify(response)

@app.route('/', methods=['GET'])
def dashboard():
    return render_template('index.html')

if __name__ == '__main__':
    # Bind to 0.0.0.0 to ensure cloud deployment accessibility (Render/Heroku/etc)
    app.run(host='0.0.0.0', port=5000)
