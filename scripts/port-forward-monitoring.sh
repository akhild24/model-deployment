#!/usr/bin/env bash

# =============================================================================
# ML Platform Monitoring Port-Forward Script
# Port-forwards Prometheus, Grafana, and Alertmanager to localhost in background.
# =============================================================================

set -euo pipefail

NAMESPACE="monitoring"

# Kill existing port-forwards for these ports to prevent conflicts
echo "Cleaning up any existing port-forwards on ports 3000, 9090, and 9093..."
pkill -f "port-forward.*3000" || true
pkill -f "port-forward.*9090" || true
pkill -f "port-forward.*9093" || true
sleep 1

echo "=== Starting background port-forwards ==="

# 1. Grafana -> localhost:3000
echo "Port-forwarding Grafana to http://localhost:3000 ..."
kubectl port-forward -n ${NAMESPACE} svc/prometheus-grafana 3000:80 > /dev/null 2>&1 &

# 2. Prometheus -> localhost:9090
echo "Port-forwarding Prometheus to http://localhost:9090 ..."
kubectl port-forward -n ${NAMESPACE} svc/prometheus-kube-prometheus-prometheus 9090:9090 > /dev/null 2>&1 &

# 3. Alertmanager -> localhost:9093
echo "Port-forwarding Alertmanager to http://localhost:9093 ..."
kubectl port-forward -n ${NAMESPACE} svc/prometheus-kube-prometheus-alertmanager 9093:9093 > /dev/null 2>&1 &

sleep 2

echo "============================================================================="
echo "🚀 Active Port-forwards:"
echo "-----------------------------------------------------------------------------"
echo "📊 Grafana:      http://localhost:3000  (Credentials: admin / admin123)"
echo "🔥 Prometheus:   http://localhost:9090"
echo "🚨 Alertmanager: http://localhost:9093"
echo "============================================================================="
echo "Keep this terminal open to maintain the port-forwards."
echo "Press Ctrl+C to terminate all port-forwards."

# Wait for background jobs to keep the script running foreground
wait
