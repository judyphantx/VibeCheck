from flask import Flask, request, jsonify
from google import genai
from google.genai import types
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  # Allow CORS for all origins

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
model = "gemini-2.0-flash"

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("message")
    print("Received message:", user_input)  # Debugging log

    if user_input.lower() == "exit":
        return jsonify({"message": "Exiting the program. Goodbye!"})

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text="""You are a finance and stock market expert. Your job is to break down whether a stock is overhyped and if it's a good buy, based on real market trends and reliable news. Use a Gen-Z tone, relatable, clear, and concise. Keep it short and sweet. No overexplaining. Emojis are necessary but don't overdo it. Give a short reasoning to why the rating is the way it is. Structure your response like this:
Quick Stock News Recap 📰 – One paragraph on the latest relevant news/trends about the stock.
Buy Rating (out of 10) 📈 – How good of a buy it is right now. Give credible news on this rating.
Hype Rating (out of 10) 🗣️ – How overhyped or underrated it is. Give credible news on this rating.
The Vibe Check 🤔 – Your call: Buy, Hold, or Sell.
Should You Buy? – Short guidance depending on risk tolerance and belief in the company.
Receipts 🧾 – A few key facts to back up your rating.
Disclaimer ⚠️ – "This content is for informational and entertainment purposes only and does not constitute financial, investment, or legal advice. Always do your own research and consult with a licensed financial advisor before making any investment decisions."
"""),
            ],
        ),
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_input)],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(response_mime_type="text/plain")

    response_text = ""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        response_text += chunk.text

    print("Bot response:", response_text)  # Debugging log

    return jsonify({"message": response_text})

if __name__ == '__main__':
    app.run(debug=True)
