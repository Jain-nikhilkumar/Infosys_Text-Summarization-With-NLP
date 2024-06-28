import pandas as pd
import re
from nltk.tokenize import sent_tokenize
import nltk
import logging
import matplotlib.pyplot as plt
from collections import Counter
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
from rouge_score import rouge_scorer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download necessary NLTK data
nltk.download('punkt')

def load_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Dataset loaded successfully from {file_path}")
        return df
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        return None

def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'\s+', ' ', text)     # Replace multiple spaces with a single space
    text = text.lower()                  # Convert to lowercase
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove non-alphanumeric characters
    return text

def preprocess_dataset(df):
    if 'article' not in df.columns:
        logging.error("'article' column is missing from the dataset")
        return df

    df = df.dropna(subset=['article'])  
    df['sentences'] = df['article'].apply(sent_tokenize)  # Tokenize sentences before cleaning
    df['cleaned_text'] = df['article'].apply(clean_text)  # Clean text
    logging.info("Dataset preprocessed successfully")
    return df

def plot_article_length_distribution(df):
    df['article_length'] = df['article'].apply(lambda x: len(x.split()))
    plt.figure(figsize=(10, 6))
    plt.hist(df['article_length'], bins=50, color='blue', edgecolor='black')
    plt.title('Distribution of Article Lengths')
    plt.xlabel('Article Length (number of words)')
    plt.ylabel('Frequency')
    plt.show()
    logging.info("Article length distribution plotted successfully")

def plot_sentence_length_distribution(df):
    df['sentence_lengths'] = df['sentences'].apply(lambda x: [len(sentence.split()) for sentence in x])
    sentence_lengths = [length for sublist in df['sentence_lengths'] for length in sublist]
    plt.figure(figsize=(10, 6))
    plt.hist(sentence_lengths, bins=50, color='green', edgecolor='black')
    plt.title('Distribution of Sentence Lengths')
    plt.xlabel('Sentence Length (number of words)')
    plt.ylabel('Frequency')
    plt.show()
    logging.info("Sentence length distribution plotted successfully")

def plot_most_common_words(df):
    all_words = df['cleaned_text'].apply(lambda x: x.split()).tolist()
    all_words = [word for sublist in all_words for word in sublist]
    word_counts = Counter(all_words)
    most_common_words = word_counts.most_common(20)
    words, counts = zip(*most_common_words)
    plt.figure(figsize=(10, 6))
    plt.bar(words, counts, color='red')
    plt.title('Most Common Words')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.show()
    logging.info("Most common words plotted successfully")

def tokenize_data(df, tokenizer, max_input_length=512, max_output_length=150):
    inputs = tokenizer(
        df['cleaned_text'].tolist(), 
        max_length=max_input_length, 
        padding='max_length', 
        truncation=True, 
        return_tensors='pt'
    )
    labels = tokenizer(
        df['highlights'].tolist(), 
        max_length=max_output_length, 
        padding='max_length', 
        truncation=True, 
        return_tensors='pt'
    )
    inputs['labels'] = labels['input_ids']
    return inputs

class SummarizationDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        return item

def generate_summary(model, tokenizer, text, max_length=150):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
    summary_ids = model.generate(
        inputs['input_ids'], 
        max_length=max_length, 
        min_length=30, 
        length_penalty=2.0, 
        num_beams=4,   # Increased num_beams for better results
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def calculate_rouge_scores(reference_summaries, generated_summaries):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for ref, gen in zip(reference_summaries, generated_summaries):
        scores = scorer.score(ref, gen)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)

    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores)
    avg_rougeL = sum(rougeL_scores) / len(rougeL_scores)

    return avg_rouge1, avg_rouge2, avg_rougeL

def main():
    # Load validation dataset
    validation_df = load_dataset('validation.csv')
    if validation_df is not None:
        
        logging.info("Validation Dataset:")
        logging.info(validation_df.head())

        validation_df = preprocess_dataset(validation_df)

        logging.info("\nValidation Dataset after cleaning and tokenization:")
        logging.info(validation_df[['article', 'cleaned_text', 'sentences']].head())

        plot_article_length_distribution(validation_df)
        plot_sentence_length_distribution(validation_df)
        plot_most_common_words(validation_df)

        # Use a smaller subset of the data for faster training
        validation_df = validation_df.sample(n=100, random_state=42)

        # Load tokenizer and tokenize data
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')  # Use smaller BART tokenizer
        inputs = tokenize_data(validation_df, tokenizer, max_input_length=256, max_output_length=128)

        # Create dataset and dataloader
        dataset = SummarizationDataset(inputs)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)  # Reduce batch size to 2

        # Load model
        model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')  # Use smaller BART model

        # Define training arguments with increased epochs and other parameters
        training_args = TrainingArguments(
            output_dir='./results',          # Output directory
            num_train_epochs=20,             # Increase the number of training epochs
            per_device_train_batch_size=2,   # Reduce batch size for training
            per_device_eval_batch_size=2,    # Reduce batch size for evaluation
            warmup_steps=500,                # Number of warmup steps for learning rate scheduler
            weight_decay=0.01,               # Strength of weight decay
            logging_dir='./logs',            # Directory for storing logs
            logging_steps=10,
            save_total_limit=1,              # Limit the total amount of checkpoints
            fp16=True,                       # Enable mixed precision training
            learning_rate=5e-5               # Adjusted learning rate
        )

        # Train model
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset
        )

        trainer.train()

        # Save model and tokenizer
        model.save_pretrained('./saved_model')
        tokenizer.save_pretrained('./saved_model')

        # Generate summaries and calculate ROUGE scores
        reference_summaries = validation_df['highlights'].tolist()
        generated_summaries = [generate_summary(model, tokenizer, text) for text in validation_df['cleaned_text'].tolist()]

        avg_rouge1, avg_rouge2, avg_rougeL = calculate_rouge_scores(reference_summaries, generated_summaries)

        print(f'ROUGE-1: {avg_rouge1:.4f}')
        print(f'ROUGE-2: {avg_rouge2:.4f}')
        print(f'ROUGE-L: {avg_rougeL:.4f}')

if __name__ == "__main__":
    main()
