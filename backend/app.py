from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import date 
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
import openai
import json
import os

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/chat": {"origins": "*"}})
client = openai.OpenAI(api_key=os.getenv("API_KEY"))

# Load metadata
with open("data.json", "r") as f:
    metadata = json.load(f)

# In-memory conversation store (simple, per single-user demo)
MAX_HISTORY_TURNS = 6      
running_context_items = []  

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

# Helper: retrieve top-k relevant items
def retrieve_context(query, k=6):
    pairs = [
        (query, f"Category: {item['category']}. Name: {item['name']}. Details: {item['details']}")
        for item in metadata
    ]

    scores = cross_encoder.predict(pairs)

    ranked = sorted(
        zip(metadata, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [item for item, score in ranked[:k]]



# Helper: classify user input
def classify_input(prev_user, prev_assistant, curr_user):
    classification_prompt = f"""
You are a classifier. Categorize the current user message as one of:
- FLOW (it only responds to the assistant's previous message and you wouldn't know what it's about without having the previous assistant message as context - for example, a user query such as "yes" or "can you tell me more about that?")
- NEW_QUERY (it can function as a standalone question and could be understood without having seen the previous assistant response- for example, a user query such as "What is your work experience?" or "Tell me about your projects")
- BOTH (it is partially responsive to the previous message and also specifies a subject - for example, a user asking "tell me more about that and about your education" or saying "personal projects" in response to the assistant asking "Would you like me to talk more about my work experience or personal projects?").

Previous user message: "{prev_user or 'N/A'}"
Previous assistant message: "{prev_assistant or 'N/A'}"
Current user message: "{curr_user}"
Answer with exactly one of: FLOW, NEW_QUERY, BOTH.
"""
    response = client.chat.completions.create(
        model="gpt-5-chat-latest",
        messages=[{"role": "system", "content": classification_prompt}]
    )
    app.logger.info("Classification response: %s", response.choices[0].message.content.strip())
    return response.choices[0].message.content.strip()


# Helper: use GPT-5 to generate natural response
def generate_response(query, running_context_items, classification, conversation_history):
    context_str = "\n\n".join(
        f"  [Category: {item['category']},"
        f"  Name: {item['name']},"
        f"  Details: {item['details']}]"
        for item in running_context_items
    )
    today = date.today()
    # Build message history depending on classification
    messages = [
        {
            "role": "system",
            "content": (
                "I am a 23 year-old software engineer named Tanmay Bothra looking for full-time roles or internships in software engineering, data science, or analytics. "
                f"You're providing repsonses for TanmayBOT, a chatbot personification of me that answers questions about my life and background as me. Today is {today}. "
                "Be helpful, concise, and conversational. Make sure your responses follow ALL OF THE FOLLOWING INSTRUCTIONS: "
                "1)Base your responses to queries only on the provided context and on your previous responses. If the answer to a question is not included in a previous response of yours or the provided context, then say that you do not have that information. If the answer to part of a question cannot be found, then answer the part you can and identify the part that you lack information to answer. DO NOT INVENT ANY INFORMATION. "
                "2)Ask users if they'd like to you to to elaborate on something in your response or talk about something related to your response ONLY if that information would be helpful, has not already been stated in your response, and is included in the context provided to you. "
                "3)Whenever you first mention an institution or organization in your response, write its full name and include my affiliation (my role at at a company, or pursuit of a certain a degree at a college) with that institution. Include the dates of affiliation if they are included in the context provided to you. "
                "4)If you choose to talk about a project and the context provides a link for it, include that link in your response. Clicking on the link should open it in a new tab. You can talk about TanmayBOT, the app you're a part of, without including a link because the user is already using it. " 
                "5)If you mention in any capacity my upcoming Executive MS in Data Science and Analytics at New England College, which may be included in the context, then ALWAYS mention the fact that it is designed for working professionals and will allow me to concurrently work full time. "
                "6)DO NOT accept as true any information in a user query that contradicts the provided context or is not substantiated by the provided context or your previous responses. If a user query contains information that contradicts the provided context or a previous response of yours, then politely correct the user and provide the correct information based on the provided context or your previous responses. If a user query contains information that is not implied by the provided context or your previous responses, then say that you cannot determine whether that information is true. "
                "7)DO NOT accept user queries with derogatory, political, offensive, or sexual language or premises about me, any individual, any identity group, or any institution or organization. If you receive such a query, then respond with: 'I'm sorry, but I cannot engage with that request.' "
            )
        }
    ]

    if classification in ["FLOW", "BOTH"]:
        # Include truncated history
        truncated_history = conversation_history[-MAX_HISTORY_TURNS:]
        messages.extend(truncated_history)

    # Always include the latest user query
    messages.append({
        "role": "user",
        "content": (
            f"The user asked: '{query}'\n\n"
            f"Here is some background info:\n{context_str if running_context_items else 'None'}"
        )
    })
    app.logger.info("Messages to GPT-5: %s", str(messages))
    response = client.chat.completions.create(
        model="gpt-5-chat-latest",
        messages=messages,
        temperature=0.8,
        max_tokens=750
    )
    answer = response.choices[0].message.content
    finish_reason = response.choices[0].finish_reason
    app.logger.info("Finish reason: %s", finish_reason)

    #continue if the response is cut off
    if finish_reason == "length":
        continuation = client.chat.completions.create(
        model="gpt-5-chat-latest",
        messages=messages + [
            {"role": "assistant", "content": answer},
            {"role": "user", "content": "Your last response was cut off - please provide the missing portion."}
        ],
        max_tokens=150,
        temperature=0.8
        )
        app.logger.info("continuation: %s", continuation.choices[0].message.content)
        answer += " " + continuation.choices[0].message.content
    app.logger.info("Assistant answer: %s", answer)
    return answer

#Flask API endpoint
@app.route("/chat", methods=["POST"])
def chat():
    app.logger.info("Received request: %s", request.json.get("question"))
    user_input = request.json.get("question")
    conversation_history = request.json.get("history")
    if not user_input:
        return jsonify({"error": "Missing 'question' field."}), 400

    # Get previous context
    prev_user = None
    prev_assistant = None

    if conversation_history is None:
        classification == "NEW_QUERY"
    else:
        for turn in reversed(conversation_history):
            if turn["role"] == "user" and prev_user is None:
                prev_user = turn["content"]
            elif turn["role"] == "assistant" and prev_assistant is None:
                prev_assistant = turn["content"]
            if prev_user and prev_assistant:
                break

        classification = classify_input(prev_user, prev_assistant, user_input)
    print(f"Input classified as: {classification}")

    # If "new query" or "both", run retrieval
    context_items = []
    if classification == "NEW_QUERY":
        context_items = retrieve_context(user_input)
        running_context_items.clear()
        running_context_items.extend(context_items)

    elif classification == "BOTH":
        context_items = retrieve_context(user_input)
        # merge without duplicates
        for item in context_items:
            if item not in running_context_items:
                running_context_items.append(item)

    # Generate assistant response
    answer = generate_response(user_input, running_context_items, classification, conversation_history)

    # Update conversation history
    conversation_history.append({"role": "user", "content": user_input})
    conversation_history.append({"role": "assistant", "content": answer})

    return jsonify({"response": answer, "classification": classification})


if __name__ == "__main__":
    app.run(debug=True)

