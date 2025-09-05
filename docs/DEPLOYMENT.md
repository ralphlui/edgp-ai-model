# Deployment Guide

## Overview

This guide covers deploying the EDGP AI Model microservice across different environments, from local development to production deployment on AWS, Azure, or Google Cloud Platform.

## Prerequisites

### System Requirements

- **Python**: 3.9 or higher
- **Docker**: 20.10+ and Docker Compose 2.0+
- **Memory**: Minimum 4GB RAM (8GB+ recommended for production)
- **Storage**: 10GB+ available disk space
- **Network**: Internet access for LLM API calls

### Required Accounts

- **AWS Account**: For Bedrock access (primary LLM provider)
- **OpenAI Account**: For fallback LLM access (optional)
- **Database**: PostgreSQL 12+ instance
- **Cache**: Redis 6+ instance

## Environment Setup

### 1. Local Development

#### Clone and Setup

```bash
# Clone repository
git clone <repository-url>
cd edgp-ai-model

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt
```

#### Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env file
vim .env
```

**Required Environment Variables**:

```bash
# Application
ENV=development
DEBUG=true
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8000

# Database
DATABASE_URL=postgresql://localhost:5432/edgp_ai
REDIS_URL=redis://localhost:6379

# AWS Bedrock (Primary LLM)
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=us-east-1
BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0

# OpenAI (Fallback)
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4

# Vector Store
VECTOR_STORE_TYPE=chromadb
CHROMA_PERSIST_DIR=./data/chroma

# Security
JWT_SECRET_KEY=your-256-bit-secret-key
API_KEY_SALT=your-api-key-salt
```

#### Start Local Services

```bash
# Start PostgreSQL and Redis with Docker
docker-compose up -d postgres redis

# Start the application
python main.py
```

### 2. Docker Development

#### Build and Run

```bash
# Build the image
docker build -t edgp-ai-model .

# Run with docker-compose
docker-compose up --build
```

#### Docker Compose Services

```yaml
# docker-compose.yml includes:
# - app: Main EDGP AI service
# - postgres: PostgreSQL database
# - redis: Redis cache
# - nginx: Reverse proxy (production)
```

### 3. Testing the Deployment

```bash
# Health check
curl http://localhost:8000/health

# Test policy suggestion endpoint
curl -X POST "http://localhost:8000/api/v1/agents/policy-suggestion" \
  -H "Content-Type: application/json" \
  -d '{
    "business_context": "Test company",
    "compliance_requirements": ["GDPR"],
    "suggestion_type": "validation_policies"
  }'
```

## Production Deployment

### AWS Deployment

#### 1. ECS with Fargate

**Task Definition** (`aws/task-definition.json`):

```json
{
  "family": "edgp-ai-model",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/edgp-ai-task-role",
  "containerDefinitions": [
    {
      "name": "edgp-ai-app",
      "image": "your-account.dkr.ecr.region.amazonaws.com/edgp-ai-model:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "ENV", "value": "production"},
        {"name": "DATABASE_URL", "value": "postgresql://rds-endpoint:5432/edgp_ai"}
      ],
      "secrets": [
        {"name": "AWS_ACCESS_KEY_ID", "valueFrom": "arn:aws:secretsmanager:region:account:secret:edgp-ai/aws-credentials"},
        {"name": "OPENAI_API_KEY", "valueFrom": "arn:aws:secretsmanager:region:account:secret:edgp-ai/openai-key"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/edgp-ai-model",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

**Deployment Script**:

```bash
#!/bin/bash
# deploy-aws.sh

# Build and push Docker image
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin your-account.dkr.ecr.us-east-1.amazonaws.com
docker build -t edgp-ai-model .
docker tag edgp-ai-model:latest your-account.dkr.ecr.us-east-1.amazonaws.com/edgp-ai-model:latest
docker push your-account.dkr.ecr.us-east-1.amazonaws.com/edgp-ai-model:latest

# Register task definition
aws ecs register-task-definition --cli-input-json file://aws/task-definition.json

# Update service
aws ecs update-service --cluster edgp-ai-cluster --service edgp-ai-service --force-new-deployment
```

#### 2. Infrastructure as Code (Terraform)

**main.tf**:

```hcl
# VPC and Networking
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  
  name = "edgp-ai-vpc"
  cidr = "10.0.0.0/16"
  
  azs             = ["us-east-1a", "us-east-1b"]
  public_subnets  = ["10.0.1.0/24", "10.0.2.0/24"]
  private_subnets = ["10.0.101.0/24", "10.0.102.0/24"]
  
  enable_nat_gateway = true
  enable_vpn_gateway = false
}

# ECS Cluster
resource "aws_ecs_cluster" "edgp_ai" {
  name = "edgp-ai-cluster"
  
  capacity_providers = ["FARGATE"]
  default_capacity_provider_strategy {
    capacity_provider = "FARGATE"
  }
}

# RDS PostgreSQL
resource "aws_db_instance" "edgp_ai" {
  identifier = "edgp-ai-db"
  
  engine         = "postgres"
  engine_version = "14.9"
  instance_class = "db.t3.micro"
  
  allocated_storage     = 20
  max_allocated_storage = 100
  storage_type          = "gp2"
  storage_encrypted     = true
  
  db_name  = "edgp_ai"
  username = "edgp_admin"
  password = var.db_password
  
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.edgp_ai.name
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  deletion_protection = true
  skip_final_snapshot = false
}

# ElastiCache Redis
resource "aws_elasticache_subnet_group" "edgp_ai" {
  name       = "edgp-ai-cache-subnet"
  subnet_ids = module.vpc.private_subnets
}

resource "aws_elasticache_cluster" "edgp_ai" {
  cluster_id           = "edgp-ai-cache"
  engine               = "redis"
  node_type            = "cache.t3.micro"
  num_cache_nodes      = 1
  parameter_group_name = "default.redis7"
  port                 = 6379
  subnet_group_name    = aws_elasticache_subnet_group.edgp_ai.name
  security_group_ids   = [aws_security_group.redis.id]
}

# Application Load Balancer
resource "aws_lb" "edgp_ai" {
  name               = "edgp-ai-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets           = module.vpc.public_subnets
  
  enable_deletion_protection = true
}

# ECS Service
resource "aws_ecs_service" "edgp_ai" {
  name            = "edgp-ai-service"
  cluster         = aws_ecs_cluster.edgp_ai.id
  task_definition = aws_ecs_task_definition.edgp_ai.arn
  desired_count   = 2
  
  capacity_provider_strategy {
    capacity_provider = "FARGATE"
    weight           = 100
  }
  
  network_configuration {
    security_groups = [aws_security_group.ecs_tasks.id]
    subnets         = module.vpc.private_subnets
  }
  
  load_balancer {
    target_group_arn = aws_lb_target_group.edgp_ai.arn
    container_name   = "edgp-ai-app"
    container_port   = 8000
  }
}
```

### Azure Deployment

#### Azure Container Instances

```bash
# Create resource group
az group create --name edgp-ai-rg --location eastus

# Create container registry
az acr create --resource-group edgp-ai-rg --name edgpairegistry --sku Basic

# Build and push image
az acr build --registry edgpairegistry --image edgp-ai-model:latest .

# Deploy container instance
az container create \
  --resource-group edgp-ai-rg \
  --name edgp-ai-app \
  --image edgpairegistry.azurecr.io/edgp-ai-model:latest \
  --cpu 2 \
  --memory 4 \
  --registry-login-server edgpairegistry.azurecr.io \
  --registry-username $(az acr credential show --name edgpairegistry --query username -o tsv) \
  --registry-password $(az acr credential show --name edgpairegistry --query passwords[0].value -o tsv) \
  --dns-name-label edgp-ai-unique \
  --ports 8000 \
  --environment-variables \
    ENV=production \
    API_PORT=8000 \
  --secure-environment-variables \
    DATABASE_URL=$DATABASE_URL \
    AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
```

### Google Cloud Platform

#### Cloud Run Deployment

```bash
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/edgp-ai-model

# Deploy to Cloud Run
gcloud run deploy edgp-ai-service \
  --image gcr.io/PROJECT_ID/edgp-ai-model \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --concurrency 100 \
  --max-instances 10 \
  --set-env-vars ENV=production \
  --set-env-vars API_PORT=8080 \
  --set-secrets DATABASE_URL=edgp-database-url:latest \
  --set-secrets AWS_ACCESS_KEY_ID=aws-access-key:latest
```

## Database Setup

### PostgreSQL Schema Migration

```bash
# Install Alembic for migrations
pip install alembic

# Initialize Alembic
alembic init alembic

# Create migration
alembic revision --autogenerate -m "Initial schema"

# Apply migration
alembic upgrade head
```

### Initial Data Setup

```sql
-- Create database
CREATE DATABASE edgp_ai;

-- Create application user
CREATE USER edgp_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE edgp_ai TO edgp_user;

-- Connect to database
\c edgp_ai;

-- Create schema
CREATE SCHEMA IF NOT EXISTS edgp;

-- Grant permissions
GRANT ALL ON SCHEMA edgp TO edgp_user;
```

## Security Configuration

### 1. API Security

#### JWT Configuration

```python
# core/security.py
from jose import JWTError, jwt
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt
```

#### API Key Management

```python
# core/auth.py
async def verify_api_key(api_key: str = Depends(get_api_key)):
    """Verify API key from database or cache"""
    # Check Redis cache first
    cached_key = await redis_client.get(f"api_key:{api_key}")
    if cached_key:
        return json.loads(cached_key)
    
    # Check database
    db_key = await get_api_key_from_db(api_key)
    if db_key:
        # Cache for 1 hour
        await redis_client.setex(f"api_key:{api_key}", 3600, json.dumps(db_key))
        return db_key
    
    raise HTTPException(status_code=401, detail="Invalid API key")
```

### 2. Infrastructure Security

#### AWS IAM Policy

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream"
      ],
      "Resource": [
        "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0",
        "arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-text-express-v1"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:us-east-1:account:log-group:/ecs/edgp-ai-model:*"
    }
  ]
}
```

## Load Balancing & Auto-scaling

### AWS Application Load Balancer

```bash
# Create target group
aws elbv2 create-target-group \
  --name edgp-ai-targets \
  --protocol HTTP \
  --port 8000 \
  --vpc-id vpc-12345678 \
  --health-check-path /health \
  --health-check-interval-seconds 30 \
  --healthy-threshold-count 2 \
  --unhealthy-threshold-count 3

# Create load balancer
aws elbv2 create-load-balancer \
  --name edgp-ai-alb \
  --subnets subnet-12345678 subnet-87654321 \
  --security-groups sg-12345678
```

### Auto-scaling Configuration

```bash
# Create auto-scaling target
aws application-autoscaling register-scalable-target \
  --service-namespace ecs \
  --scalable-dimension ecs:service:DesiredCount \
  --resource-id service/edgp-ai-cluster/edgp-ai-service \
  --min-capacity 2 \
  --max-capacity 10

# Create scaling policy
aws application-autoscaling put-scaling-policy \
  --service-namespace ecs \
  --scalable-dimension ecs:service:DesiredCount \
  --resource-id service/edgp-ai-cluster/edgp-ai-service \
  --policy-name edgp-ai-scaling-policy \
  --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration '{
    "TargetValue": 70.0,
    "PredefinedMetricSpecification": {
      "PredefinedMetricType": "ECSServiceAverageCPUUtilization"
    },
    "ScaleOutCooldown": 300,
    "ScaleInCooldown": 300
  }'
```

## Monitoring & Logging

### 1. Application Monitoring

#### CloudWatch Configuration

```python
# core/monitoring.py
import boto3
from datetime import datetime

cloudwatch = boto3.client('cloudwatch')

def put_metric(metric_name: str, value: float, unit: str = 'Count'):
    """Send custom metrics to CloudWatch"""
    cloudwatch.put_metric_data(
        Namespace='EDGP/AI/Model',
        MetricData=[
            {
                'MetricName': metric_name,
                'Value': value,
                'Unit': unit,
                'Timestamp': datetime.utcnow()
            }
        ]
    )

# Usage in agents
put_metric('AgentRequests', 1, 'Count')
put_metric('ResponseTime', response_time_ms, 'Milliseconds')
put_metric('TokensUsed', token_count, 'Count')
```

#### Custom Metrics

- **Agent Performance**: Response times, success rates
- **LLM Usage**: Token consumption, cost tracking
- **Data Quality**: Quality scores, anomaly detection rates
- **System Health**: Memory usage, CPU utilization

### 2. Structured Logging

```python
# core/logging_config.py
import structlog
import logging.config

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "()": structlog.stdlib.ProcessorFormatter,
            "processor": structlog.dev.ConsoleRenderer(colors=False),
            "foreign_pre_chain": [
                structlog.stdlib.add_log_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.ExtraAdder(),
                structlog.processors.TimeStamper(fmt="iso"),
            ],
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "json",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "logs/edgp-ai.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "formatter": "json",
        },
    },
    "loggers": {
        "edgp_ai": {
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": False,
        },
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
logger = structlog.get_logger("edgp_ai")
```

## Database Migration

### Production Migration Strategy

```python
# migrations/migrate.py
import asyncio
from sqlalchemy import create_engine
from alembic.config import Config
from alembic import command

async def run_migrations():
    """Run database migrations in production"""
    
    # 1. Backup database
    await backup_database()
    
    # 2. Run migrations in transaction
    alembic_cfg = Config("alembic.ini")
    command.upgrade(alembic_cfg, "head")
    
    # 3. Verify migration success
    await verify_schema()
    
    # 4. Update application configuration
    await update_app_config()

async def rollback_migration(target_revision: str):
    """Rollback to specific migration"""
    alembic_cfg = Config("alembic.ini")
    command.downgrade(alembic_cfg, target_revision)
```

## Backup & Recovery

### 1. Database Backup

```bash
#!/bin/bash
# backup-db.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="edgp_ai_backup_${DATE}.sql"

# Create backup
pg_dump $DATABASE_URL > $BACKUP_FILE

# Upload to S3
aws s3 cp $BACKUP_FILE s3://edgp-ai-backups/database/

# Keep local backups for 7 days
find . -name "edgp_ai_backup_*.sql" -mtime +7 -delete
```

### 2. Application State Backup

```python
# core/backup.py
async def backup_vector_store():
    """Backup vector store data"""
    # Export embeddings and metadata
    pass

async def backup_redis_cache():
    """Backup Redis cache data"""
    # Export cache data for faster recovery
    pass
```

## Performance Optimization

### 1. Caching Strategy

```python
# core/cache.py
import redis.asyncio as redis
from typing import Optional, Any
import json

class CacheManager:
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        value = await self.redis.get(key)
        return json.loads(value) if value else None
    
    async def set(self, key: str, value: Any, ttl: int = 3600):
        """Set cached value with TTL"""
        await self.redis.setex(key, ttl, json.dumps(value))
    
    async def invalidate_pattern(self, pattern: str):
        """Invalidate keys matching pattern"""
        keys = await self.redis.keys(pattern)
        if keys:
            await self.redis.delete(*keys)
```

### 2. Database Connection Pooling

```python
# core/database.py
from sqlalchemy.pool import QueuePool
from sqlalchemy import create_engine

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True,
    pool_recycle=3600
)
```

## Disaster Recovery

### 1. Multi-Region Setup

```bash
# Primary region: us-east-1
# Secondary region: us-west-2

# Cross-region database replication
aws rds create-db-instance-read-replica \
  --db-instance-identifier edgp-ai-db-replica \
  --source-db-instance-identifier edgp-ai-db \
  --destination-region us-west-2
```

### 2. Failover Procedures

```python
# core/failover.py
async def health_check_and_failover():
    """Monitor primary services and failover if needed"""
    
    # Check primary database
    if not await check_database_health():
        await switch_to_replica_database()
    
    # Check primary LLM provider
    if not await check_llm_health():
        await switch_to_fallback_llm()
    
    # Notify administrators
    await send_failover_alert()
```

## Cost Optimization

### 1. LLM Cost Management

```python
# core/cost_optimization.py
class CostOptimizer:
    def __init__(self):
        self.model_costs = {
            "anthropic.claude-3-sonnet": 0.003,  # per 1K tokens
            "anthropic.claude-3-haiku": 0.0015,
            "gpt-4": 0.03,
            "gpt-3.5-turbo": 0.002
        }
    
    def select_optimal_model(self, task_complexity: str, budget_limit: float):
        """Select most cost-effective model for task"""
        if task_complexity == "simple" and budget_limit < 0.01:
            return "anthropic.claude-3-haiku"
        elif task_complexity == "complex":
            return "anthropic.claude-3-sonnet"
        else:
            return "gpt-3.5-turbo"
```

### 2. Resource Scaling

```python
# Auto-scaling based on queue depth
async def scale_based_on_queue():
    queue_depth = await get_queue_depth()
    
    if queue_depth > 100:
        await scale_up_instances(target_count=queue_depth // 50)
    elif queue_depth < 10:
        await scale_down_instances(min_count=2)
```

## Troubleshooting

### Common Issues

#### 1. LLM Connection Errors

```bash
# Check AWS credentials
aws sts get-caller-identity

# Test Bedrock access
aws bedrock list-foundation-models --region us-east-1

# Check model availability
aws bedrock get-foundation-model --model-identifier anthropic.claude-3-sonnet-20240229-v1:0
```

#### 2. Database Connection Issues

```bash
# Test database connection
psql $DATABASE_URL -c "SELECT version();"

# Check connection pool
curl http://localhost:8000/debug/db-pool-status
```

#### 3. Performance Issues

```bash
# Check memory usage
docker stats

# Check logs for bottlenecks
docker logs edgp-ai-app | grep "SLOW_QUERY\|ERROR"

# Monitor metrics
curl http://localhost:8000/metrics | jq '.response_times'
```

### Debug Mode

```bash
# Enable debug mode
export DEBUG=true
export LOG_LEVEL=DEBUG

# Start with debug logging
python main.py --debug
```

## Maintenance

### 1. Regular Tasks

```bash
#!/bin/bash
# maintenance.sh

# Update Python dependencies
pip-review --local --auto

# Clean old logs
find logs/ -name "*.log" -mtime +30 -delete

# Optimize database
psql $DATABASE_URL -c "VACUUM ANALYZE;"

# Clear old cache entries
redis-cli FLUSHDB
```

### 2. Model Updates

```python
# core/model_manager.py
async def update_models():
    """Update to latest model versions"""
    
    # Check for new Bedrock models
    new_models = await check_available_models()
    
    # Test new models with sample data
    for model in new_models:
        await test_model_performance(model)
    
    # Update configuration if performance improved
    await update_model_config(best_performing_model)
```

## Deployment Checklist

### Pre-deployment

- [ ] Environment variables configured
- [ ] Database migrations tested
- [ ] AWS/Cloud permissions verified
- [ ] SSL certificates installed
- [ ] Monitoring setup completed
- [ ] Backup procedures tested

### Post-deployment

- [ ] Health checks passing
- [ ] Logs aggregating correctly
- [ ] Metrics being collected
- [ ] Performance within acceptable ranges
- [ ] Error rates below threshold
- [ ] Security scans completed

### Rollback Plan

- [ ] Previous version tagged and available
- [ ] Database rollback scripts prepared
- [ ] Load balancer traffic routing configured
- [ ] Monitoring alerts configured
- [ ] Communication plan for downtime

This deployment guide provides comprehensive instructions for deploying the EDGP AI Model across various environments with proper security, monitoring, and maintenance procedures.
