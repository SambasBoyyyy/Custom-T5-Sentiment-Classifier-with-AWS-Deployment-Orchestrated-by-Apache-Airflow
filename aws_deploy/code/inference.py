import json
import torch
import os
import logging
from transformers import T5Tokenizer
# Import the custom model class. 
# Ensure t5_sentiment_gate.py is in the same directory (code/) inside the tarball.
try:
    from t5_sentiment_gate import T5ForSentimentClassification
except ImportError:
    # Fallback or helpful error if the file isn't found during local testing
    import sys
    sys.path.append(os.path.dirname(__file__))
    from t5_sentiment_gate import T5ForSentimentClassification

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def model_fn(model_dir, context=None):
    """
    Load the model for inference
    """
    logger.info(f"Loading model from {model_dir}")
    if context:
        logger.info(f"Context provided: {context}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer
    # The artifacts are in model_dir
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    
    # Load model
    # We assume the custom class has a from_pretrained method compatible with HuggingFace
    model = T5ForSentimentClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    
    logger.info("Model loaded successfully")
    return {"model": model, "tokenizer": tokenizer}

def input_fn(request_body, request_content_type):
    """
    Deserialize the request body
    """
    if request_content_type == "application/json":
        data = json.loads(request_body)
        if "text" in data:
            return data["text"]
        if "inputs" in data:
            return data["inputs"]
        return data
    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model_dict):
    """
    Generate predictions
    """
    try:
        model = model_dict["model"]
        tokenizer = model_dict["tokenizer"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Input data: {input_data}")
        
        # Tokenize input
        inputs = tokenizer(input_data, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        
        with torch.no_grad():
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            logger.info(f"Model output type: {type(outputs)}")
            logger.info(f"Model output keys: {outputs.keys() if isinstance(outputs, dict) else 'Not a dict'}")
            
            # Handle output. Assuming classification logits.
            # The custom model returns a dict {'logits': ...}
            if isinstance(outputs, dict):
                if "logits" in outputs:
                    logits = outputs["logits"]
                else:
                    # Fallback, maybe it's just the logits
                    logger.warning("Output dict does not contain 'logits'. Using output as is.")
                    logits = outputs
            elif hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs[0]
                
            logger.info(f"Logits shape: {logits.shape}")
                
            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)
            
            # Get the predicted class (0 or 1)
            score, prediction_idx = torch.max(probs, dim=-1)
            
            # Map to labels
            # 0 -> negative, 1 -> positive
            label_map = {0: "negative", 1: "positive"}
            prediction_label = label_map.get(prediction_idx.item(), "unknown")
            
            result = {
                "label": prediction_label,
                "score": float(score.item())
            }
            logger.info(f"Prediction result: {result}")
            return result
    except Exception as e:
        logger.error(f"Error in predict_fn: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return a dummy error response so we can see it in the API response if possible
        # Or re-raise with a better message
        raise ValueError(f"Prediction failed: {str(e)}")

def output_fn(prediction, response_content_type):
    """
    Serialize the prediction result
    """
    if response_content_type == "application/json":
        return json.dumps(prediction)
    raise ValueError(f"Unsupported content type: {response_content_type}")
