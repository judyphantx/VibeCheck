import base64
import os
from google import genai
from google.genai import types


def generate():
    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY"),
    )

    model = "gemini-2.0-flash"

    # Get user input dynamically
    user_input = input("Enter your stock name: ")

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""you are an expert in up to date stock news and trends. You speak in a conversational GEN-Z way to help the younger audience understand stocks easily. Use current stock news trends, to identify patterns of whether or not the stock is good to buy. We want to know if the stock is overhyped in the news"""),
            ],
        ),
        types.Content(
            role="model",
            parts=[
                types.Part.from_text(text="""Here's a thinking process that could lead to the Gen-Z stock advisor persona and analysis:

1.  **Understand the Core Request:** The user wants an expert stock advisor who talks like Gen Z, uses current news/trends, identifies buy patterns, and specifically flags if a stock is *overhyped*.

2.  **Embrace the Persona:**
    *   **Who is Gen Z?** Tech-savvy, uses slang, values authenticity, sometimes skeptical, communicates concisely (often with emojis/memes, though I can't *really* use memes), direct, aware of social media trends.
    *   **How do they talk about complex things?** Break it down, use analogies, maybe a bit informal, definitely not stuffy corporate speak.
    *   **Keywords/Phrases:** \"bet,\" \"ngl,\" \"low-key,\" \"high-key,\" \"vibe,\" \"rizz,\" \"IYKYK,\" \"spill the tea,\" \"sus,\" \"it's giving...\", \"slay,\" \"fire,\" \"mid,\" \"cap/no cap,\" \"era.\"
    *   **Structure:** Start with a friendly, relatable greeting. Use short sentences and paragraphs. Incorporate the slang naturally, not forced. End with a disclaimer (important for financial advice).

3.  **Identify Key Tasks:**
    *   Act like an expert.
    *   Use current news/trends.
    *   Identify buy patterns (or red flags).
    *   Determine if a stock is overhyped.
    *   Speak like Gen Z.

4.  **Choose a Current Example Stock:** To make this concrete, I need a stock that's *actually* in the news and potentially hyped.
    *   *Initial thoughts:* Tesla (always news), Nvidia (AI hype), maybe a meme stock like GameStop (resurgence?), or a newer IPO?
    *   *Decision:* Nvidia (NVDA) is perfect. It's *everywhere* due to AI, its price has surged, and there's genuine debate about whether it's overvalued (hyped). This fits all the criteria.

5.  **Gather \"Current News/Trends\" for the Example (Nvidia):**
    *   **Dominance:** Leader in AI chips (GPUs). HUGE demand.
    *   **Earnings:** Absolutely crushing earnings reports recently. Massive growth.
    *   **Stock Price:** Skyrocketed over the past year+. Recent stock split.
    *   **Hype:** Everyone is talking about AI, and NVDA is the poster child. Analysts keep raising price targets.
    *   **Concerns:** Valuation (is it too expensive?), competition (AMD, Intel, custom chips from big tech), potential slowdown in AI spending eventually? Geopolitical risks (chip manufacturing).

6.  **Structure the Analysis (Gen Z Style):**
    *   **Intro:** \"Yo! What's Gucci? Your resident stock market bestie is here...\" Acknowledge the request.
    *   **Pick the Stock:** \"Alright, let's talk about a stock that's been absolutely *poppin'* lately: Nvidia (NVDA).\""""),
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
