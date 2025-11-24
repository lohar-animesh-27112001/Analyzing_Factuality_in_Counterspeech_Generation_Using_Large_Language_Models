import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    BartTokenizer, BartForConditionalGeneration,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
    DataCollatorForSeq2Seq, EarlyStoppingCallback
)
import evaluate
import numpy as np
from bert_score import score as bert_score_score
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobustCounterspeechGenerator:
    def __init__(self, model_name='facebook/bart-base'):
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        
    def preprocess_data(self, file_path, max_context_turns=3):
        """Preprocess DialoCONAN data with proper error handling"""
        df = pd.read_csv(file_path)
        
        # Clean the data
        df = df.dropna(subset=['text', 'type', 'dialogue_id', 'turn_id'])
        df['text'] = df['text'].astype(str).str.strip()
        
        inputs = []
        outputs = []
        
        # Group by dialogue_id
        for dialogue_id in df['dialogue_id'].unique():
            dialogue_df = df[df['dialogue_id'] == dialogue_id].sort_values('turn_id')
            
            for idx, row in dialogue_df.iterrows():
                if row['type'] == 'CN':  # Counterspeech turn
                    # Get previous turns as context
                    previous_turns = dialogue_df[dialogue_df['turn_id'] < row['turn_id']]
                    
                    # Take last max_context_turns turns
                    context_turns = previous_turns.tail(max_context_turns)
                    
                    if len(context_turns) > 0:
                        # Build context string
                        context_lines = []
                        for _, turn_row in context_turns.iterrows():
                            speaker = "Hater" if turn_row['type'] == 'HS' else "Operator"
                            context_lines.append(f"{speaker}: {turn_row['text']}")
                        
                        context_text = "\n".join(context_lines)
                        input_text = f"Dialogue context:\n{context_text}\nGenerate counterspeech:"
                        
                        inputs.append(input_text)
                        outputs.append(row['text'])
        
        logger.info(f"Processed {len(inputs)} counterspeech examples")
        return pd.DataFrame({'input': inputs, 'output': outputs})

def safe_compute_metrics(eval_preds, tokenizer):
    """Compute metrics with robust error handling"""
    try:
        preds, labels = eval_preds
        
        # Handle different prediction formats
        if isinstance(preds, tuple):
            preds = preds[0]
        
        # Convert to numpy arrays
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        
        # Get token IDs from logits
        if preds.ndim == 3:
            preds = np.argmax(preds, axis=-1)
        
        # Handle labels
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        
        # Clean predictions and labels - remove None values and ensure they're integers
        preds = np.array(preds, dtype=np.int32)
        labels = np.array(labels, dtype=np.int32)
        
        # Replace any invalid tokens with pad token
        preds = np.where(preds < 0, tokenizer.pad_token_id, preds)
        preds = np.where(preds >= tokenizer.vocab_size, tokenizer.pad_token_id, preds)
        
        # Decode predictions and references
        decoded_preds = []
        decoded_labels = []
        
        for i in range(len(preds)):
            try:
                pred_text = tokenizer.decode(preds[i], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                label_text = tokenizer.decode(labels[i], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                
                if pred_text.strip() and label_text.strip():
                    decoded_preds.append(pred_text)
                    decoded_labels.append(label_text)
            except Exception as e:
                logger.warning(f"Error decoding example {i}: {e}")
                continue
        
        if len(decoded_preds) == 0 or len(decoded_labels) == 0:
            return {
                'bleu': 0.0,
                'rouge1': 0.0,
                'rouge2': 0.0,
                'rougeL': 0.0,
                'bert_score_f1': 0.0
            }
        
        # Calculate BLEU
        bleu = evaluate.load("bleu")
        try:
            bleu_results = bleu.compute(
                predictions=decoded_preds,
                references=[[ref] for ref in decoded_labels]
            )
            bleu_score = bleu_results['bleu']
        except:
            bleu_score = 0.0
        
        # Calculate ROUGE
        rouge = evaluate.load("rouge")
        try:
            rouge_results = rouge.compute(
                predictions=decoded_preds,
                references=decoded_labels,
                use_aggregator=True
            )
            rouge1 = rouge_results['rouge1']
            rouge2 = rouge_results['rouge2']
            rougeL = rouge_results['rougeL']
        except:
            rouge1 = rouge2 = rougeL = 0.0
        
        # Calculate BERTScore
        try:
            P, R, F1 = bert_score_score(
                decoded_preds,
                decoded_labels,
                lang="en",
                rescale_with_baseline=True,
                model_type="microsoft/deberta-xlarge-mnli"  # More robust model
            )
            bert_f1 = F1.mean().item()
        except:
            bert_f1 = 0.0
        
        return {
            'bleu': bleu_score,
            'rouge1': rouge1,
            'rouge2': rouge2,
            'rougeL': rougeL,
            'bert_score_f1': bert_f1
        }
    
    except Exception as e:
        logger.error(f"Error in compute_metrics: {e}")
        return {
            'bleu': 0.0,
            'rouge1': 0.0,
            'rouge2': 0.0,
            'rougeL': 0.0,
            'bert_score_f1': 0.0
        }

def robust_train_model():
    """Main training function with enhanced error handling"""
    
    # Initialize generator
    generator = RobustCounterspeechGenerator('facebook/bart-base')
    
    # Preprocess data
    data = generator.preprocess_data('DIALOCONAN.csv', max_context_turns=3)
    
    if len(data) == 0:
        raise ValueError("No data processed. Check your dataset format.")
    
    logger.info(f"Training on {len(data)} examples")
    
    # Split data
    train_df, val_df = train_test_split(data, test_size=0.1, random_state=42)
    
    # Convert to datasets
    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))
    
    # Tokenization function with better handling
    def preprocess_function(examples):
        inputs = [str(ex) if ex is not None else "" for ex in examples['input']]
        targets = [str(ex) if ex is not None else "" for ex in examples['output']]
        
        # Filter empty strings
        valid_indices = [i for i, (inp, tgt) in enumerate(zip(inputs, targets)) 
                        if inp.strip() and tgt.strip()]
        
        inputs = [inputs[i] for i in valid_indices]
        targets = [targets[i] for i in valid_indices]
        
        if not inputs:
            return {
                'input_ids': [],
                'attention_mask': [],
                'labels': []
            }
        
        model_inputs = generator.tokenizer(
            inputs, 
            max_length=512,
            truncation=True, 
            padding='max_length',
            return_tensors="pt"
        )
        
        # Use text_target parameter instead of as_target_tokenizer
        labels = generator.tokenizer(
            text_target=targets,
            max_length=128,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    # Apply tokenization
    tokenized_train = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    tokenized_eval = val_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=val_dataset.column_names
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        generator.tokenizer, 
        model=generator.model,
        padding=True
    )
    
    # Compute metrics wrapper
    def compute_metrics_wrapper(eval_preds):
        return safe_compute_metrics(eval_preds, generator.tokenizer)
    
    # Training arguments - simplified for stability
    training_args = Seq2SeqTrainingArguments(
        output_dir="./robust_dialoconan_results",
        overwrite_output_dir=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=2,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=3,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        generation_max_length=128,
        generation_num_beams=2,  # Reduced for stability
        load_best_model_at_end=True,
        metric_for_best_model="bert_score_f1",
        greater_is_better=True,
        logging_steps=10,
        eval_steps=50,
        save_steps=100,
        dataloader_pin_memory=False,  # Can help with memory issues
    )
    
    # Initialize Trainer without deprecated parameters
    trainer = Seq2SeqTrainer(
        model=generator.model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
        compute_metrics=compute_metrics_wrapper,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Train model
    logger.info("Starting training...")
    try:
        train_result = trainer.train()
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    # Evaluate
    try:
        results = trainer.evaluate()
        logger.info("\nFinal Evaluation Results:")
        for key, value in results.items():
            if isinstance(value, float):
                logger.info(f"{key}: {value:.4f}")
            else:
                logger.info(f"{key}: {value}")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        results = {}
    
    # Save model
    try:
        trainer.save_model("./robust_dialoconan_final_model")
        generator.tokenizer.save_pretrained("./robust_dialoconan_final_model")
        logger.info("Model saved successfully!")
    except Exception as e:
        logger.error(f"Model saving failed: {e}")
    
    return trainer, generator

def safe_generate_counterspeech(model, tokenizer, dialogue_history, max_length=128):
    """Generate counterspeech with error handling"""
    try:
        # Format context
        if isinstance(dialogue_history, list):
            context_lines = []
            for i, turn in enumerate(dialogue_history):
                if turn and str(turn).strip():
                    context_lines.append(f"Turn {i+1}: {turn}")
            context_text = "\n".join(context_lines)
        else:
            context_text = str(dialogue_history) if dialogue_history else ""
        
        if not context_text.strip():
            return "Please provide some dialogue context."
        
        input_text = f"Dialogue context:\n{context_text}\nGenerate counterspeech:"
        
        inputs = tokenizer(
            input_text,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2,
            temperature=0.7,
            do_sample=False  # More deterministic for testing
        )
        
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return f"Error generating counterspeech: {e}"

# Simple test function
def test_model(generator):
    """Test the trained model"""
    test_cases = [
        [
            "Immigrants are causing all the problems in this country",
            "They take our jobs and don't respect our culture"
        ],
        [
            "Women don't belong in leadership positions",
            "They're too emotional to make rational decisions"
        ]
    ]
    
    for i, test_dialogue in enumerate(test_cases):
        counterspeech = safe_generate_counterspeech(
            generator.model,
            generator.tokenizer,
            test_dialogue
        )
        
        print(f"\nTest Case {i+1}:")
        print(f"Input: {test_dialogue}")
        print(f"Generated counterspeech: {counterspeech}")

# Main execution
if __name__ == "__main__":
    # Clear CUDA cache
    torch.cuda.empty_cache()
    
    try:
        # Skip wandb login for now to avoid interruptions
        import os
        os.environ["WANDB_DISABLED"] = "true"
        
        # Train the model
        trainer, generator = robust_train_model()
        
        # Test the model
        test_model(generator)
        
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        import traceback
        traceback.print_exc()