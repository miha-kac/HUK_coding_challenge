from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification


model = AutoModelForSequenceClassification.from_pretrained("./HUK_DistilBert")
tokenizer = AutoTokenizer.from_pretrained("./HUK_DistilBert")


def predict_sentiment(text):
    output_dict = {0: 'Negative', 1: 'Positive', 2: 'Neutral', 3: 'Irrelevant'}
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    predictions = outputs.logits.argmax(dim=-1)
    result = predictions.item()
    return output_dict.get(result)


app = Flask(__name__)
@app.route('/')
def home():
    return "Welcome to the HUK Sentiment Analyzer!"


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    text = data.get('text', '')
    sentiment = predict_sentiment(text)
    return jsonify({'Sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)
