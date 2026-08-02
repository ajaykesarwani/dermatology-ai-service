#!/usr/bin/env bash
# =============================================================================
# deploy_aws.sh — Deploy DermDiagnostic AI to AWS ECS Fargate via ECR
# =============================================================================
# Usage:
#   chmod +x aws/deploy_aws.sh
#   ./aws/deploy_aws.sh              # interactive confirmation
#   ./aws/deploy_aws.sh --dry-run    # L2 FIX: print all steps without executing
#
# Prerequisites:
#   - AWS CLI installed & configured (aws configure)
#   - Docker installed
#   - jq installed (brew install jq / apt install jq)
#
# Environment variables (set these or export them beforehand):
#   AWS_REGION       (default: eu-west-1)
#   AWS_ACCOUNT_ID   (auto-detected if not set)
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# L2 FIX — Dry-run mode: print every step without executing any destructive
# commands (no docker build/push, no ECS task registration).
# ---------------------------------------------------------------------------
DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "⚠️  DRY-RUN MODE — no changes will be made to AWS or Docker."
fi

run() {
    # Wrapper: echoes the command, then executes it (or skips in dry-run)
    echo "  + $*"
    if [[ "$DRY_RUN" == false ]]; then
        "$@"
    fi
}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
AWS_REGION="${AWS_REGION:-eu-west-1}"
APP_NAME="dermatology-ai"
ECR_REPO="${APP_NAME}-api"
IMAGE_TAG="$(git rev-parse --short HEAD)"

AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  DermDiagnostic AI — AWS Deployment Script"
echo "  Region  : ${AWS_REGION}"
echo "  Account : ${AWS_ACCOUNT_ID}"
echo "  Image   : ${ECR_URI}:${IMAGE_TAG}"
echo "  Dry-run : ${DRY_RUN}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# L2 FIX — Confirmation prompt before making any production changes
if [[ "$DRY_RUN" == false ]]; then
    read -r -p "⚠️  This will push a new image and update the production ECS service. Continue? [y/N] " confirm
    if [[ ! "${confirm}" =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

# ── Step 1: Ensure ECR repository exists ──────────────────────────────────────
echo "[1/5] Ensuring ECR repository '${ECR_REPO}' exists..."
run aws ecr describe-repositories \
    --repository-names "${ECR_REPO}" \
    --region "${AWS_REGION}" > /dev/null 2>&1 || \
run aws ecr create-repository \
    --repository-name "${ECR_REPO}" \
    --image-scanning-configuration scanOnPush=true \
    --region "${AWS_REGION}"

# ── Step 2: Authenticate Docker to ECR ───────────────────────────────────────
echo "[2/5] Authenticating Docker to ECR..."
run bash -c "aws ecr get-login-password --region '${AWS_REGION}' | \
    docker login --username AWS --password-stdin '${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com'"

# ── Step 3: Build and push Docker image ───────────────────────────────────────
echo "[3/5] Building Docker image..."
run docker build \
    -t "${ECR_REPO}:${IMAGE_TAG}" \
    -t "${ECR_REPO}:latest" \
    -f server/Dockerfile .

echo "[3/5] Tagging for ECR..."
run docker tag "${ECR_REPO}:${IMAGE_TAG}" "${ECR_URI}:${IMAGE_TAG}"
run docker tag "${ECR_REPO}:latest"       "${ECR_URI}:latest"

echo "[3/5] Pushing to ECR..."
run docker push "${ECR_URI}:${IMAGE_TAG}"
run docker push "${ECR_URI}:latest"

# ── Step 4: Update ECS Task Definition ───────────────────────────────────────
echo "[4/5] Registering new ECS Task Definition..."
TASK_DEF_JSON=$(sed \
    -e "s|<ECR_URI>|${ECR_URI}|g" \
    -e "s|<IMAGE_TAG>|${IMAGE_TAG}|g" \
    -e "s|<AWS_REGION>|${AWS_REGION}|g" \
    -e "s|<AWS_ACCOUNT_ID>|${AWS_ACCOUNT_ID}|g" \
    aws/ecs-task-definition.json)

if [[ "$DRY_RUN" == true ]]; then
    echo "  [dry-run] Would register task definition:"
    echo "${TASK_DEF_JSON}" | head -20
    NEW_TASK_DEF="arn:aws:ecs:${AWS_REGION}:${AWS_ACCOUNT_ID}:task-definition/${APP_NAME}-task:DRY_RUN"
else
    NEW_TASK_DEF=$(echo "${TASK_DEF_JSON}" | aws ecs register-task-definition \
        --cli-input-json file:///dev/stdin \
        --region "${AWS_REGION}" \
        --query 'taskDefinition.taskDefinitionArn' \
        --output text)
fi
echo "  Registered: ${NEW_TASK_DEF}"

# ── Step 5: Update ECS Service ────────────────────────────────────────────────
echo "[5/5] Updating ECS Service to use new task definition..."
run aws ecs update-service \
    --cluster "${APP_NAME}-cluster" \
    --service  "${APP_NAME}-service" \
    --task-definition "${NEW_TASK_DEF}" \
    --force-new-deployment \
    --region "${AWS_REGION}" > /dev/null

echo ""
echo "✅ Deployment complete!"
echo "   Image    : ${ECR_URI}:${IMAGE_TAG}"
echo "   Task Def : ${NEW_TASK_DEF}"
echo "   Service  : ${APP_NAME}-service on ${APP_NAME}-cluster"
echo ""
echo "  Monitor rollout:"
echo "  aws ecs describe-services --cluster ${APP_NAME}-cluster --services ${APP_NAME}-service --region ${AWS_REGION}"
