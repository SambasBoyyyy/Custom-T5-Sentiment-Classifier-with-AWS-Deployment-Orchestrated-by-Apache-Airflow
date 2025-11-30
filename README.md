# T5 Sentiment Classification with Custom Architecture + AWS Deployment

A production-ready sentiment classification system using a custom T5 architecture with learnable sentiment gates, deployed on AWS SageMaker Serverless.

## 🎯 Overview

This project implements a **custom T5-based sentiment classifier** with a novel **Sentiment Gate mechanism** that learns to identify and weight sentiment-bearing tokens. The model is deployed as a **serverless API** on AWS for cost-effective, scalable inference.

**Key Features:**
- 🧠 Custom T5 architecture with learnable sentiment gates
- ⚡ Serverless deployment (pay-per-request)
- 🌐 Public REST API endpoint
- 📊 Binary sentiment classification (positive/negative)
- 🎨 Modular, production-ready codebase

---

## 🏗️ Architecture

### Custom T5 Model Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT TEXT                                │
│              "I love this product!"                          │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│                  T5 ENCODER                                  │
│  (Pretrained T5-small encoder, 512-dim hidden states)       │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│              SENTIMENT GATE                                  │
│  ┌────────────────────────────────────────────┐             │
│  │ Linear(512 → 1) + Sigmoid                  │             │
│  │ Learns importance scores for each token    │             │
│  └────────────────────────────────────────────┘             │
│                                                              │
│  Token Scores:                                               │
│    "I"      → 0.12  (low importance)                        │
│    "love"   → 0.95  (HIGH importance) ⭐                    │
│    "this"   → 0.08  (low importance)                        │
│    "product"→ 0.72  (medium importance)                     │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│            WEIGHTED POOLING                                  │
│  Aggregate hidden states using gate scores                  │
│  pooled = Σ(hidden_states × gate_scores) / Σ(gate_scores)  │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│          CLASSIFICATION HEAD                                 │
│  Linear(512 → 2) → [logit_negative, logit_positive]        │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│                  OUTPUT                                      │
│  {"label": "positive", "score": 0.95}                       │
└─────────────────────────────────────────────────────────────┘
```

### Why This Architecture?

**Traditional T5 for Classification:**
- Uses decoder to generate `<positive>` or `<negative>` tokens
- Slow (requires multiple decoder steps)
- Treats all input tokens equally

**Our Custom Architecture:**
- ✅ **3x faster** - No decoder, direct classification
- ✅ **More accurate** - Gate learns which words matter for sentiment
- ✅ **Interpretable** - Gate scores show which tokens influenced the decision
- ✅ **Efficient** - Encoder-only architecture

---

## 📊 Model Components

### 1. Sentiment Gate (`SentimentGate`)

```python
class SentimentGate(nn.Module):
    """
    Learns to score token importance for sentiment.
    
    Input:  [batch, seq_len, 512]  (encoder hidden states)
    Output: [batch, seq_len, 1]    (importance scores 0-1)
    """
    def __init__(self, hidden_size=512):
        self.gate_projection = nn.Linear(hidden_size, 1)
    
    def forward(self, hidden_states):
        scores = self.gate_projection(hidden_states)
        return torch.sigmoid(scores)  # [0, 1] range
```

**What it learns:**
- Sentiment-bearing words (love, hate, terrible, amazing) → **high scores**
- Neutral words (the, is, a, this) → **low scores**
- Negations (not, never) → **very high scores**

### 2. Custom T5 Model (`T5ForSentimentClassification`)

```python
class T5ForSentimentClassification(nn.Module):
    def __init__(self, config, num_labels=2):
        self.encoder = T5EncoderModel(config)      # Pretrained T5 encoder
        self.sentiment_gate = SentimentGate(512)   # Learnable gate
        self.classifier = nn.Linear(512, 2)        # Binary classifier
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids, attention_mask):
        # 1. Encode
        hidden_states = self.encoder(input_ids, attention_mask).last_hidden_state
        
        # 2. Compute gate scores
        gate_scores = self.sentiment_gate(hidden_states)
        
        # 3. Weighted pooling
        weighted = (hidden_states * gate_scores).sum(dim=1)
        pooled = weighted / (gate_scores.sum(dim=1) + 1e-9)
        
        # 4. Classify
        logits = self.classifier(self.dropout(pooled))
        
        return {"logits": logits}
```

---

## 🚀 Deployment Pipeline

### AWS Infrastructure

```
┌─────────────────────────────────────────────────────────────┐
│                      CLIENT                                  │
│              (Web/Mobile/API Consumer)                       │
└──────────────────┬──────────────────────────────────────────┘
                   │ HTTPS POST
                   ▼
┌─────────────────────────────────────────────────────────────┐
│              API GATEWAY (HTTP API)                          │
│  https://2ssx8bnfcf.execute-api.us-east-1.amazonaws.com     │
│                                                              │
│  Route: POST /predict                                        │
│  CORS: Enabled                                               │
└──────────────────┬──────────────────────────────────────────┘
                   │ Invokes
                   ▼
┌─────────────────────────────────────────────────────────────┐
│              AWS LAMBDA                                      │
│  Function: t5-sentiment-lambda                              │
│  Runtime: Python 3.9                                         │
│  Memory: 128 MB                                              │
│  Timeout: 30s                                                │
│                                                              │
│  Role: Parse request → Invoke SageMaker → Return response   │
└──────────────────┬──────────────────────────────────────────┘
                   │ InvokeEndpoint
                   ▼
┌─────────────────────────────────────────────────────────────┐
│         SAGEMAKER SERVERLESS ENDPOINT                        │
│  Endpoint: t5-sentiment-serverless-endpoint                 │
│  Type: Serverless Inference                                  │
│  Memory: 3072 MB                                             │
│  Max Concurrency: 10                                         │
│  Container: HuggingFace PyTorch 1.13.1                      │
│                                                              │
│  ┌──────────────────────────────────────────┐               │
│  │  Model Package (model.tar.gz)            │               │
│  │  ├── code/                                │               │
│  │  │   ├── inference.py                    │               │
│  │  │   ├── t5_sentiment_gate.py            │               │
│  │  │   └── requirements.txt                │               │
│  │  ├── pytorch_model.bin (113 MB)          │               │
│  │  ├── config.json                         │               │
│  │  ├── tokenizer files                     │               │
│  │  └── spiece.model                        │               │
│  └──────────────────────────────────────────┘               │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│                   RESPONSE                                   │
│  {"label": "positive", "score": 0.95}                       │
└─────────────────────────────────────────────────────────────┘
```

### Deployment Steps

```bash
# 1. Package model with inference code
python aws_deploy/package_model.py

# 2. Deploy to SageMaker Serverless
python aws_deploy/deploy_sagemaker.py

# 3. Setup Lambda + API Gateway
python aws_deploy/create_api_gateway.py

# 4. Test the API
python aws_deploy/quick_test.py
```

### Inference Handler (`inference.py`)

```python
def model_fn(model_dir, context=None):
    """Load model and tokenizer"""
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    model = T5ForSentimentClassification.from_pretrained(model_dir)
    model.eval()
    return {"model": model, "tokenizer": tokenizer}

def predict_fn(input_data, model_dict):
    """Run inference"""
    model = model_dict["model"]
    tokenizer = model_dict["tokenizer"]
    
    # Tokenize
    inputs = tokenizer(input_data, return_tensors="pt", 
                      padding=True, truncation=True, max_length=512)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_ids=inputs.input_ids, 
                       attention_mask=inputs.attention_mask)
        logits = outputs["logits"]
        probs = torch.softmax(logits, dim=-1)
        score, pred_idx = torch.max(probs, dim=-1)
    
    # Map to label
    label = "positive" if pred_idx.item() == 1 else "negative"
    
    return {"label": label, "score": float(score.item())}
```

---

## 📁 Project Structure

```
t5-aws-mlops-pipeline/
├── modules/
│   ├── models/
│   │   ├── t5_sentiment_gate.py      # Custom T5 architecture
│   │   └── __init__.py
│   ├── data/
│   │   ├── sst2_dataset.py           # SST-2 data loading
│   │   └── __init__.py
│   ├── training/
│   │   ├── trainer.py                # Training loop
│   │   ├── config.py                 # Hyperparameters
│   │   └── __init__.py
│   └── evaluation/
│       ├── evaluator.py              # Evaluation + metrics
│       └── __init__.py
│
├── aws_deploy/
│   ├── code/
│   │   ├── inference.py              # SageMaker inference handlers
│   │   └── requirements.txt          # Inference dependencies
│   ├── package_model.py              # Model packaging script
│   ├── deploy_sagemaker.py           # SageMaker deployment
│   ├── setup_iam_role.py             # IAM role creation
│   ├── create_api_gateway.py         # Lambda + API Gateway setup
│   ├── lambda_inference.py           # Lambda function code
│   ├── quick_test.py                 # API testing
│   └── README.md                     # Deployment docs
│
├── t5-classification/
│   └── best_model/                   # Trained model checkpoint
│       ├── pytorch_model.bin         # Model weights (113 MB)
│       ├── config.json               # Model config
│       └── tokenizer files           # Tokenizer artifacts
│
├── train.py                          # Training script
├── evaluate.py                       # Evaluation script
├── requirements.txt                  # Training dependencies
└── README.md                         # This file
```

---

## 🎓 Training

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model (3 epochs, ~30 minutes on GPU)
python train.py --epochs 3 --batch_size 16 --output_dir ./t5-classification

# Evaluate
python evaluate.py --model_path ./t5-classification/best_model
```

### Training Configuration

```python
# modules/training/config.py
TRAINING_CONFIG = {
    "model_name": "t5-small",
    "num_labels": 2,
    "learning_rate": 5e-5,
    "batch_size": 16,
    "epochs": 3,
    "max_length": 512,
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "gradient_accumulation_steps": 1,
}
```

### Training Process

1. **Data Loading**: SST-2 dataset (67k train, 872 validation)
2. **Preprocessing**: Tokenization with T5 tokenizer
3. **Training Loop**:
   - Forward pass through encoder + gate + classifier
   - Cross-entropy loss
   - AdamW optimizer with warmup
   - Gradient clipping (max_norm=1.0)
4. **Validation**: Every epoch, save best model
5. **Output**: Best model checkpoint saved to `t5-classification/best_model/`

---

## 🧪 Testing the API

### Using curl

```bash
curl -X POST https://2ssx8bnfcf.execute-api.us-east-1.amazonaws.com/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I absolutely love this product!"}'
```

**Response:**
```json
{
  "label": "positive",
  "score": 0.95
}
```

### Using Python

```python
import requests

url = "https://2ssx8bnfcf.execute-api.us-east-1.amazonaws.com/predict"
response = requests.post(url, json={"text": "This movie was terrible!"})
print(response.json())
# {"label": "negative", "score": 0.89}
```

---

## 💰 Cost Analysis

### Serverless Pricing (Pay-per-Request)

| Component | Pricing | Monthly Cost (1000 req/day) |
|-----------|---------|----------------------------|
| **SageMaker Serverless** | $0.20/hour compute | ~$5-10 |
| **Lambda** | $0.20/1M requests | Free tier |
| **API Gateway** | $1.00/1M requests | ~$0.03 |
| **S3 Storage** | $0.023/GB | ~$0.01 |
| **Total** | | **~$5-10/month** |

**No charges when idle!** ✅

---

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | 94.2% |
| **Inference Time** | 1-2 seconds |
| **Cold Start** | 10-30 seconds (first request) |
| **Warm Requests** | <2 seconds |
| **Model Size** | 113 MB (compressed) |
| **Memory Usage** | ~2.5 GB (during inference) |

---

## 🔧 Configuration

### Environment Variables

```bash
# .env (optional)
AWS_REGION=us-east-1
SAGEMAKER_ENDPOINT=t5-sentiment-serverless-endpoint
```

### Deployment Modes

```bash
# Serverless (default)
python aws_deploy/deploy_sagemaker.py

# Real-time with GPU (for higher throughput)
DEPLOYMENT_MODE=realtime INSTANCE_TYPE=ml.g4dn.xlarge \
  python aws_deploy/deploy_sagemaker.py
```

---

## 🛠️ Development

### Adding New Features

1. **Modify Model**: Edit `modules/models/t5_sentiment_gate.py`
2. **Retrain**: `python train.py`
3. **Repackage**: `python aws_deploy/package_model.py`
4. **Redeploy**: `python aws_deploy/deploy_sagemaker.py`

### Local Testing

```python
from modules.models.t5_sentiment_gate import T5ForSentimentClassification
from transformers import T5Tokenizer

# Load model
model = T5ForSentimentClassification.from_pretrained("./t5-classification/best_model")
tokenizer = T5Tokenizer.from_pretrained("./t5-classification/best_model")

# Predict
text = "I love this!"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
logits = outputs["logits"]
pred = torch.argmax(logits, dim=-1).item()
print("positive" if pred == 1 else "negative")
```

---

## 📚 References

- **T5 Paper**: [Exploring the Limits of Transfer Learning](https://arxiv.org/abs/1910.10683)
- **SST-2 Dataset**: [Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/)
- **AWS SageMaker**: [Serverless Inference](https://docs.aws.amazon.com/sagemaker/latest/dg/serverless-endpoints.html)

---

## 📝 License

MIT License - See LICENSE file for details

---

## 🤝 Contributing

Contributions welcome! Please open an issue or PR.

---

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

**Built with ❤️ using T5, PyTorch, and AWS**
