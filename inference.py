from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TextClassificationPipeline
import argparse
import time
import json
import os


def load_model(model_dir):
    model_class, tokenizer_class, pretrained_weights = (AutoModelForSequenceClassification, AutoTokenizer, model_dir)
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights, do_lower_case=True)
    model = model_class.from_pretrained(pretrained_weights, num_labels=6)
    return tokenizer, model


def predict(text, model_dir):
    
    ### loading prerequisites and label encoded dictionary
    tokenizer, model = load_model(model_dir)
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=False)
    with open(os.path.join(model_dir,'label_dict.json'),'r') as f:
        label_dict = json.load(f)
    label_dict = {str(v):k for (k,v) in label_dict.items()}
    
    ### prediction with model and inverse labelling the output
    text = text.lower()
    output = pipe(text)[0]
    label = output.get('label')[-1]
    return label_dict.get(label)



def arg_parser():
    parser = argparse.ArgumentParser(description="Inference for emotion label inside text")
    parser.add_argument('--text', type=str, required=True, help="Input full sentence for inference")
    parser.add_argument('--model',type=str, required=False,
                        default=r'C:\Workplace\finetuned-models\emotion-classification\distilbert-finetuned\saved',
                        help='Directory of finetuned model for inference')
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    input_text = args.text
    model_dir = args.model
    print("Input text is \x1B[3m'{}'\x1B[0m".format(input_text))           ### Italic output
    print("Infering from \x1B[3m{}\x1B[0m\n".format(model_dir))            ### Italic output
    time.sleep(2)
    print("Your input expresses \x1B[3m{}\x1B[0m".format(predict(input_text, model_dir)))


