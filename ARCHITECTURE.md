# DermDiagnostic AI — Architecture Reference

## System Overview

DermDiagnostic AI is a production-grade, end-to-end medical imaging pipeline for automated dermatological skin lesion classification, built to the standards of a commercial MedTech product.

```
┌─────────────────────────────────────────────────────────────────┐
│                     DermDiagnostic AI System                    │
│                                                                  │
│  ┌──────────────────┐        ┌──────────────────────────────┐   │
│  │   C# WPF Client  │  HTTP  │   FastAPI Inference Service  │   │
│  │  (Windows App)   │───────▶│      (Python / Docker)       │   │
│  │                  │◀───────│                              │   │
│  │  • Image Upload  │  JSON  │  • /predict  (POST)          │   │
│  │  • Confidence UI │        │  • /healthz  (GET)           │   │
│  │  • Risk Colors   │        │  • /metrics  (GET)           │   │
│  └──────────────────┘        │  • /info     (GET)           │   │
│                              └──────────────┬───────────────┘   │
│                                             │                   │
│                              ┌──────────────▼───────────────┐   │
│                              │   ONNX Runtime (INT8 Quant)  │   │
│                              │   ResNet-18 (HAM10000 FT)    │   │
│                              └──────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Preprocessing Pipeline

Each image passes through the following stages before reaching the ONNX model:

```
Raw JPEG/PNG/BMP
       │
       ▼
┌─────────────────────────────────────────┐
│  1. Dull Razor Hair Removal (OpenCV)    │
│     • Black-Hat morphological filter   │
│     • Gaussian blur + threshold mask   │
│     • Telea inpainting restoration     │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  2. Resize to 224×224 px               │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  3. BGR → RGB conversion               │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  4. Normalize to [0, 1] range          │
│     ÷ 255.0                            │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  5. ImageNet Normalization             │
│     mean = [0.485, 0.456, 0.406]       │
│     std  = [0.229, 0.224, 0.225]       │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  6. HWC → CHW format transpose         │
│  7. Add batch dimension → (1,3,224,224)│
└──────────────────┬──────────────────────┘
                   │
                   ▼
             ONNX Runtime
```

---

## Model Architecture

| Property | Value |
|---|---|
| Base Architecture | ResNet-18 |
| Training Dataset | HAM10000 (10,015 real dermoscopic images) |
| Output Classes | 7 (akiec, bcc, bkl, df, mel, nv, vasc) |
| Final Layer | `nn.Linear(512, 7)` |
| Loss Function | CrossEntropyLoss |
| Optimizer | AdamW (lr=1e-4) |
| Training Epochs | 5 |
| Export Format | ONNX opset 18 |
| Inference Format | INT8 Dynamic Quantization |
| Inference Speed | ~22ms on CPU (i7), ~5ms on GPU |

---

## Deployment Architectures

### 1. Local Development (Docker Compose)
```
Developer Machine
└── docker-compose up --build
    └── FastAPI container (port 8000)
        └── models/efficientnet_quant_int8.onnx (read-only volume)
```

### 2. NVIDIA Triton Inference Server
```
GPU Server / Cloud VM
└── docker-compose -f deployment/docker-compose.triton.yml up
    └── Triton Server (port 8000 HTTP, 8001 gRPC, 8002 metrics)
        └── Model Repository: deployment/triton/skin_lesion_classifier/
            └── config.pbtxt          (model configuration)
            └── 1/model.onnx          (versioned model weights)
```
**Benefits:** Dynamic batching, GPU TensorRT FP16 optimization, multi-model serving, concurrent execution, gRPC interface.

### 3. AWS ECS Fargate (Production)
```
                   Internet
                      │
              ┌───────▼────────┐
              │  AWS ALB       │  port 80
              └───────┬────────┘
                      │
         ┌────────────▼────────────┐
         │    AWS ECS Fargate      │
         │  ┌──────────────────┐  │
         │  │  Container 1     │  │
         │  │  (dermatology-api│  │  Deployed via:
         │  │   :latest)       │  │  • ECR image registry
         │  └──────────────────┘  │  • Terraform IaC
         │  ┌──────────────────┐  │  • deploy_aws.sh
         │  │  Container 2     │  │  • CloudWatch logs
         │  │  (dermatology-api│  │  • Circuit breaker
         │  │   :latest)       │  │    auto-rollback
         │  └──────────────────┘  │
         └─────────────────────────┘
```

---

## CI/CD Pipeline

```
git push → GitHub Actions CI
               │
               ├── [1] Lint (Ruff)
               ├── [2] Unit Tests (pytest)
               ├── [3] Docker Compose syntax validation
               └── [4] Docker Build validation
                         │
                         ▼ (on success)
                   Ready to deploy
                   via deploy_aws.sh
```

---

## File Structure

```
dermatology-ai-service/
├── server/                       # FastAPI inference microservice
│   ├── app/
│   │   ├── main.py               # API endpoints & ONNX session
│   │   └── vision/
│   │       └── preprocessing.py  # Dull Razor + normalization pipeline
│   ├── tests/
│   │   └── test_api.py           # Unit tests
│   ├── Dockerfile
│   └── requirements.txt
│
├── models/                       # Model artifacts & training
│   ├── train.py                  # PyTorch training script (HAM10000)
│   ├── export_onnx.py            # PyTorch → ONNX + INT8 quantization
│   ├── benchmark.py              # Latency benchmarking
│   └── resnet18_ham10000.pth     # Trained weights (gitignored)
│
├── client/                       # C# WPF Desktop Application
│   └── DermDiagnostic.Wpf/
│       ├── MainWindow.xaml       # UI layout
│       └── MainWindow.xaml.cs    # Inference client logic
│
├── deployment/                   # Deployment configurations
│   ├── docker-compose.yml        # Standard FastAPI deployment
│   ├── docker-compose.triton.yml # NVIDIA Triton deployment
│   ├── k8s-deployment.yaml       # Kubernetes manifest
│   └── triton/                   # Triton model repository
│       ├── skin_lesion_classifier/
│       │   └── config.pbtxt      # Triton model config
│       └── triton_client.py      # Triton HTTP inference client
│
├── aws/                          # AWS cloud infrastructure
│   ├── deploy_aws.sh             # ECR build + ECS deploy script
│   ├── ecs-task-definition.json  # ECS Fargate task config
│   └── terraform/
│       └── main.tf               # Full IaC: VPC, ALB, ECS, ECR
│
└── .github/workflows/
    └── ci.yml                    # GitHub Actions CI pipeline
```
