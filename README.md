# ml-platform

A production-grade MLOps platform serving a semantic text similarity model via a scalable, observable microservice on AWS EKS. Built as a portfolio project targeting real-world engineering practices across the full MLOps lifecycle.

---

## Overview

This platform exposes a [HuggingFace `all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) sentence-transformers model as a REST API. The primary use case is semantic similarity scoring — for example, matching resumes against job descriptions. The project emphasizes production readiness: infrastructure-as-code, CI/CD, autoscaling, and a full observability stack.

---

## Architecture

```
Client
  │
  ▼
AWS ALB (internet-facing)
  │
  ▼
EKS Cluster (us-east-1) — 2× t3.medium Spot nodes
  │
  ├── ml-platform namespace
  │     ├── FastAPI Deployment  (2–5 replicas, HPA)
  │     └── ClusterIP Service
  │
  └── monitoring namespace
        ├── Prometheus
        ├── Grafana
        ├── Loki + Promtail
        └── Alertmanager
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **API** | FastAPI, Pydantic, Uvicorn |
| **Model** | HuggingFace sentence-transformers (`all-MiniLM-L6-v2`) |
| **Containerization** | Docker (multi-stage, weights baked in) |
| **Registry** | AWS ECR |
| **Infrastructure** | Terraform (dev/prod workspaces), AWS VPC, EKS, NAT Gateway, IAM/OIDC |
| **Orchestration** | Kubernetes, Helm, AWS ALB Ingress Controller |
| **Autoscaling** | HPA (CPU 70% / Memory 80%, 2–5 replicas) |
| **CI/CD** | Jenkins on EC2 |
| **Observability** | Prometheus, Grafana, Loki, Promtail, Alertmanager |
| **Dev Environment** | WSL/Ubuntu, VS Code |

---

## Project Phases

### ✅ Phase 1 — FastAPI Application
- `/health`, `/embed`, `/similarity`, `/metrics` endpoints
- Pydantic request/response validation
- Prometheus instrumentation via `prometheus_fastapi_instrumentator`
- Custom `Histogram` metric: `embedding_inference_latency_seconds`
- pytest suite with mocked model inference

### ✅ Phase 2 — Docker & ECR
- Multi-stage Dockerfile with model weights baked in at build time
- `.dockerignore` for lean image builds
- Makefile for common build/push workflows
- Image pushed to AWS ECR

### ✅ Phase 3 — Terraform Infrastructure
- VPC with public/private subnets across `us-east-1a` and `us-east-1b`
- NAT Gateway for private subnet egress
- EKS cluster (k8s 1.30) with managed node group (Spot instances)
- IAM roles, OIDC provider, IRSA for ALB Controller
- Dev/prod workspace separation

### ✅ Phase 4 — Kubernetes & Ingress
- Namespace, ConfigMap, Deployment (2 replicas), ClusterIP Service
- AWS ALB Ingress Controller (via Helm, IRSA-wired)
- Internet-facing ALB with subnet auto-discovery
- HPA: 2–5 replicas, CPU and memory thresholds
- `metrics-server` installed for HPA resource metrics

### ✅ Phase 5 — Jenkins CI/CD
- Jenkins on EC2 (Amazon Corretto Java 21)
- Pipeline stages: Checkout → Build Docker → Push ECR → `kubectl set image` → Rollout status
- Jenkins IAM role mapped to `system:main` in EKS `aws-auth` ConfigMap

### ✅ Phase 6 — Observability Stack
- Prometheus scrapes pods annotated with `prometheus.io/scrape: "true"` in the `ml-platform` namespace
- Grafana dashboard auto-provisioned via ConfigMap sidecar watcher
- Loki (single-binary, cost-optimized) + Promtail for log aggregation
- Alertmanager alert rules:
  - `HighInferenceLatency`: p95 > 2s for 5 min
  - `HighErrorRate`: 5xx > 5% for 5 min
  - `PodDown`: available replicas < desired for 2 min
- Access via port-forward (private network model)

---

## Repository Structure

```
ml-platform/
├── app/
│   ├── main.py              # FastAPI app, endpoints, Prometheus metrics
│   └── model.py             # sentence-transformers wrapper
├── infra/
│   └── terraform/           # VPC, EKS, IAM, OIDC
├── k8s/
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── ingress.yaml
│   ├── hpa.yaml
│   └── monitoring/          # Prometheus, Grafana, Loki ConfigMaps
├── scripts/
│   ├── setup-monitoring.sh  # Idempotent observability stack deploy
│   └── port-forward-monitoring.sh
├── tests/
│   └── test_api.py
├── Dockerfile
├── Makefile
└── Jenkinsfile
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/embed` | Generate sentence embedding |
| `POST` | `/similarity` | Cosine similarity between two texts |
| `GET` | `/metrics` | Prometheus metrics scrape endpoint |

### Example: Similarity Request

```bash
curl -X POST https://<ALB_ENDPOINT>/similarity \
  -H "Content-Type: application/json" \
  -d '{
    "text1": "Experienced Python developer with AWS and Kubernetes skills",
    "text2": "Looking for a backend engineer with cloud infrastructure experience"
  }'
```

```json
{
  "similarity": 0.847,
  "inference_time_ms": 23.4
}
```

---

## Infrastructure

### AWS Resources

| Resource | Value |
|---|---|
| EKS Cluster | `ml-platform-dev-cluster` |
| Region | `us-east-1` |
| Node type | `t3.medium` Spot, ×2 |
| Kubernetes version | `1.30` |

### Accessing the Observability Stack

```bash
# Start all port-forwards
./scripts/port-forward-monitoring.sh

# Grafana:      http://localhost:3000  (admin / admin123)
# Prometheus:   http://localhost:9090
# Alertmanager: http://localhost:9093
```

---

## Local Development

### Prerequisites
- Python 3.10+, Docker, `kubectl`, `helm`, `terraform`, AWS CLI
- WSL/Ubuntu recommended on Windows

### Run Locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
# API available at http://localhost:8000
```

### Run Tests

```bash
pytest tests/ -v
```

### Build & Push Docker Image

```bash
make build
make push
```

---

## Infrastructure Management

```bash
# Provision
cd infra/terraform
terraform workspace select dev
terraform apply

# Tear down (cost control)
terraform destroy
```

> **Note:** ALB Controller must be reinstalled via Helm after every `terraform apply` before applying `ingress.yaml`. Always pass `--set vpcId=` and `--set region=` explicitly — auto-detection via EC2 instance metadata can silently fail.

---

## Key Engineering Decisions

**Model weights baked into the image** — eliminates cold-start latency from remote model downloads; trades image size for startup speed.

**Spot instances with HPA** — Spot instances cut compute costs significantly. HPA ensures availability during Spot interruptions by scaling up replacements before nodes drain.

**IRSA over node-level IAM** — AWS Load Balancer Controller uses pod-level identity via OIDC/IRSA, scoped to `system:serviceaccount:kube-system:aws-load-balancer-controller`. Reduces blast radius vs. broad node IAM roles.

**Private observability access** — Prometheus and Grafana are not exposed via ALB. Port-forwarding keeps the observability plane off the public internet without the overhead of VPN or private endpoints.

**Idempotent monitoring setup** — `setup-monitoring.sh` uses `helm upgrade --install` and is safe to re-run across `terraform destroy/apply` cycles.

---

## Known Limitations & Interview Notes

This project demonstrates core MLOps platform engineering. Production systems would additionally require:

- **Model registry** (e.g., MLflow, SageMaker Model Registry) for versioning and lineage
- **Training pipelines** for retraining on new data
- **Data drift monitoring** to detect when the input distribution diverges from training data
- **Canary deployments** for safer model rollouts with gradual traffic shifting
- **Private Grafana access** via ALB with authentication rather than port-forwarding

---

## License

MIT
