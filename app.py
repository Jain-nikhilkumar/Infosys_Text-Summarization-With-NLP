import gradio as gr
from transformers import BartForConditionalGeneration, BartTokenizer

# Load saved model and tokenizer
saved_model_directory = './saved_model'

tokenizer = BartTokenizer.from_pretrained(saved_model_directory)
model = BartForConditionalGeneration.from_pretrained(saved_model_directory)

def summarize_text(article):
    inputs = tokenizer(article, return_tensors='pt', max_length=512, truncation=True)
    summary_ids = model.generate(
        inputs['input_ids'],
        max_length=150,
        min_length=30,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

interface = gr.Interface(
    fn=summarize_text,
    inputs=gr.Textbox(lines=10, label="Article Text"),
    outputs=gr.Textbox(label="Summary"),
    title="Text Summarization",
    description="Enter the article text to generate a summary."
)

if __name__ == "__main__":
    interface.launch()
