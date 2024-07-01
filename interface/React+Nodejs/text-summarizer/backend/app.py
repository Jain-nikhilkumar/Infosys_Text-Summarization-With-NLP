

from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load model and tokenizer
model_name = "google/pegasus-xsum"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    text = data['text']
    inputs = tokenizer(text, truncation=True, padding='longest', return_tensors="pt")
    # Adjust the parameters for better summarization
    summary_ids = model.generate(
        inputs.input_ids, 
        max_length=250, 
        min_length=30, 
        length_penalty=2.0, 
        num_beams=4, 
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return jsonify({'summary': summary})

if __name__ == '__main__':
    app.run(debug=True)
