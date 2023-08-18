import numpy as np
from bert_score import BERTScorer
import evaluate  # Assuming this module contains the load function for loading the 'rouge' object

# Load BERT scorer with specified parameters
bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    # Decode the predicted and target sequences
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute ROUGE scorest
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    # Compute BERTScore metrics
    P, R, F1 = bert_scorer.score(decoded_preds, decoded_labels)
    bertscore_metrics = {
        'bertscore_precision': P.mean().item(),
        'bertscore_recall': R.mean().item(),
        'bertscore_f1': F1.mean().item()
    }

    # Compute average generated sequence length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result.update(bertscore_metrics)
    result["gen_len"] = np.mean(prediction_lens)

    # Round the results for presentation
    return {k: round(v, 4) for k, v in result.items()}
