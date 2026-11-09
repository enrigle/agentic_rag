# Intent

## Problem

I keep my notes, repos, courses, and references in Notion. I want to ask questions about all of them in plain language. Notion AI can do this, but I must pay for it.

The chatbot gives bad answers to three types of questions:

- **List and topic questions.** Examples: "Give me my skills." "Give me all information about Claude." Only 5 chunks go to the LLM. Thus, the LLM cannot list many items.
- **Follow-up questions.** Example: "Give me the skills" after a previous answer. The search does not get the context of the conversation.
- **"Do I have information on X?" questions.** The answer does not clearly say yes or no. The answer does not give the pages.

## Outcome

A chatbot that knows all my Notion content. It replaces Notion AI for me.

## Users

Only me. One person on one computer.

## Constraints

- The chatbot must not cost money. Free cloud LLMs (Groq) are permitted. Ollama is the local fallback.
- Notion is the only data source. The chatbot only reads Notion.
- Answers come from my Notion. Web search operates only when I set it to on.
- The workspace is small: about 69 pages and 1,300 chunks. Design for this size.

## Non-goals

- Skills or files on my computer. Only Notion content is in scope.
- A graph database.
- Changes to Notion content.
- More than one user.
- Live sync. Sync when the app starts, or when I start it.
- A mobile app.

## Success check

1. Write 15–20 real questions. Claude writes the questions. I write the correct answers.
2. Include list and topic questions, "do I have X" questions, and follow-up questions.
3. Run `scripts/eval.py`. All questions must pass.
4. Use the chatbot every day. I must not need Notion AI.

## Open questions

- A router decides where each question goes. Groq is the router now. Does Groq make too many errors? If yes, change the router to Jev. The test results give the answer.
