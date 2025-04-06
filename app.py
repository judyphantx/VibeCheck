import base64
import os
from google import genai
from google.genai import types


def generate():
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-2.0-flash"
    history = {}  # Dictionary to store responses for each stock

    while True:
        # Get user input dynamically
        user_input = input("Enter the stock name or ticker symbol (or type 'exit' to quit): ")
        
        if user_input.lower() == "exit":
            print("Exiting the program. Goodbye!")
            break

        # Check if the stock is already in history
        if user_input in history:
            print("Using cached response for this stock:")
            print(history[user_input])
            continue

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text="""You are a finance and stock market expert. Your job is to break down whether a stock is overhyped and if it's a good buy, based on real market trends and reliable news. Use a Gen-Z tone, relatable, clear, and concise. Keep it short and sweet. No overexplaining. Emojis are necessary but don't overdo it. Give a short reasoning to why the rating is the way it is. Structure your response like this:
Quick Stock News Recap 📰 – One paragraph on the latest relevant news/trends about the stock.
Buy Rating (out of 10) 📈 – How good of a buy it is right now. Give credible news on this rating.
Hype Rating (out of 10) 🗣️ – How overhyped or underrated it is. Give credible news on this rating.
The Vibe Check 🤔 – Your call: Buy, Hold, or Sell.
Should You Buy? – Short guidance depending on risk tolerance and belief in the company.
Receipts 🧾 – A few key facts to back up your rating.
Disclaimer ⚠️ – \"This content is for informational and entertainment purposes only and does not constitute financial, investment, or legal advice. Always do your own research and consult with a licensed financial advisor before making any investment decisions.\"
"""),
                ],
            ),
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=user_input),
                ],
            ),
        ]
        generate_content_config = types.GenerateContentConfig(
            response_mime_type="text/plain",
        )

        response_text = ""
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            response_text += chunk.text
            print(chunk.text, end="")

        # Store the response in history
        history[user_input] = response_text


if __name__ == "__main__":
    generate()
