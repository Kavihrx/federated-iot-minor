from flask import Flask, jsonify, request
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
    if is_anomalous:
        # Anomalous data: higher variance, shifted mean
        X = np.random.randn(50, NUM_FEATURES) * 2 + 1.5
        y = np.ones(50)
    else:
        # Normal data
        X = np.random.randn(50, NUM_FEATURES)
        y = np.zeros(50)
    return X, y

def simulate_llm_classification(anomaly_score):
    """
    Simulates a lightweight LLM-inspired classification module.
    In the paper, an LLM would process the textual anomaly logs.
    Here, we use a simple heuristic based on the anomaly score to classify the attack type.
    """
    if anomaly_score < 0.5:
        return "Normal Traffic"
    elif 0.5 <= anomaly_score < 0.8:
        return "Possible Reconnaissance Activity"
    else:
        return "Intrusion Detected: DDoS"

@app.route('/api/federated_pipeline', methods=['GET', 'POST'])
def run_pipeline():
    client_updates_weights = []
    client_updates_bias = []
    client_anomaly_logs = []
    
    # --- PHASE 1: Client Local Training & Anomaly Detection ---
    for i in range(NUM_CLIENTS):
        # Randomly decide if this client experiences anomalous traffic
        experiences_anomaly = random.random() > 0.6 
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
        final_alerts.append({
            "status": simulate_llm_classification(0.1),
            "details": "No anomalies detected across all clients."
        })
    else:
        for log in client_anomaly_logs:
            classification = simulate_llm_classification(log['anomaly_score'])
            final_alerts.append({
                "client_id": log['client_id'],
                "status": classification,
                "anomaly_score": log['anomaly_score']
            })
            
    # Return aggregated result with the classification outputs
    response = {
        "message": "Federated rounds completed successfully.",
        "federated_server_status": "Weights Aggregated (FedAvg)",
        "results": final_alerts
    }
    
    return jsonify(response)

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        "status": "online", 
        "service": "Federated LLM-Driven Intrusion Detection Aggregation Server",
        "usage": "Access /api/federated_pipeline to trigger a simulation round."
    })

if __name__ == '__main__':
    # Bind to 0.0.0.0 to ensure cloud deployment accessibility (Render/Heroku/etc)
    app.run(host='0.0.0.0', port=5000)
