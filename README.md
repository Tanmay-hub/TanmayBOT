[TanmayBOT](https://tanmaybot.vercel.app/) is an AI chatbot personification of me that can answer questions about my work experience, skills, projects, education, and interests.

The *tanmaybot-frontend* folder contains is React.js app used to build the UI, and the *backend* folder includes data.json, which is a knowledge base with details about me, and a Flask app. The Flask API uses a cross-encoder model from the sentence_transformers library to fetch the portions of the knowledge base most relevant to a user query and then submits them as context along with the user query to GPT-5, which generates responses.

