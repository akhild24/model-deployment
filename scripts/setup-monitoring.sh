#!/usr/bin/env bash

# =============================================================================
# ML Platform Monitoring Stack Setup Script
# Installs Prometheus, Grafana, Alertmanager, Loki, and Promtail on AWS EKS.
# =============================================================================

set -euo pipefail

NAMESPACE="monitoring"
APP_NAMESPACE="ml-platform"

echo "=== [1/5] Creating Namespace: ${NAMESPACE} ==="
kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -

echo "=== [2/5] Adding Helm Repositories ==="
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

echo "=== [3/5] Installing kube-prometheus-stack (Prometheus, Grafana, Alertmanager) ==="
# Relative paths are based on running from the project root
helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
  --namespace ${NAMESPACE} \
  --values k8s/monitoring/prometheus-values.yaml

echo "=== [4/5] Installing Loki-stack (Loki & Promtail) ==="
helm upgrade --install loki grafana/loki-stack \
  --namespace ${NAMESPACE} \
  --values k8s/monitoring/loki-values.yaml

echo "=== [5/5] Deploying Custom Grafana Dashboards ==="
kubectl apply -f k8s/monitoring/grafana-dashboards-configmap.yaml

echo "=== Waiting for Monitoring Stack Pods to be Ready ==="
echo "Waiting for prometheus-prometheus deployment to spin up..."
kubectl rollout status statefulset/prometheus-prometheus-kube-prometheus-prometheus -n ${NAMESPACE} --timeout=180s || true

echo "Waiting for Grafana deployment..."
kubectl rollout status deployment/prometheus-grafana -n ${NAMESPACE} --timeout=180s

echo "============================================================================="
echo "🎉 Monitoring stack successfully installed!"
echo "============================================================================="
echo "Grafana Dashboard: ML Platform Overview"
echo "Username: admin"
echo "Password: admin123"
echo ""
echo "To access the stack locally, run the port-forward script:"
echo "  ./scripts/port-forward-monitoring.sh"
echo "============================================================================="
