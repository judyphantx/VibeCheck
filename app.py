import base64
import os
from google import genai
from google.genai import types


def generate():
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-2.0-flash"

    while True:
        # Get user input dynamically
        user_input = input("Enter the stock name or ticker symbol (or type 'exit' to quit): ")
        
        if user_input.lower() == "exit":
            print("Exiting the program. Goodbye!")
            break

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
                role="model",
                parts=[
                    types.Part.from_text(text="""Alright, bet. Let's break down this stock.

**Quick Stock News Recap 📰**

Okay, so Palantir (PLTR) has been kinda popping off lately. They're all about that AI life, and they've been snagging some big government and enterprise contracts. Earnings reports have been decent, but the stock price has been kinda wild, swinging up and down like crazy. They are now included in the S&P 500 index. 🥳

**Buy Rating (out of 10) 📈:** 6/10

*   **Why:** Palantir is a polarizing stock. While it is in the S&P 500, valuation metrics are very high, and it is up a lot this year. Be cautious when dealing with speculative companies.

**Hype Rating (out of 10) 🗣️:** 8/10

*   **Why:** Palantir is definitely having a moment. Everyone's talking about it, especially with the AI buzz. Some think it's gonna revolutionize everything, others think it's just a fad. There are lots of discussions across social media platforms, and it tends to be a trending stock. The stock price has more than doubled this year alone. 🚀

**The Vibe Check 🤔:** Hold

**Should You Buy?**

*   **High Risk:** If you're feeling risky and believe in the long-term AI play, maybe a small position.
*   **Low Risk:** Probably sit this one out or wait for a dip.
*   **Already holding:** Hold.

**Receipts 🧾**

*   Big government contracts = $$$
*   AI hype = potentially unsustainable price
*   Volatile stock price = 🎢

**Disclaimer ⚠️**

This content is for informational and entertainment purposes only and does not constitute financial, investment, or legal advice. Always do your own research and consult with a licensed financial advisor before making any investment decisions.
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

        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            print(chunk.text, end="")

if __name__ == "__main__":
    generate()
