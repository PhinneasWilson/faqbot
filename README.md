# faqbot
An offline FAQ chatbot that retrieves answers using pure-Python TF-IDF cosine similarity.

## Setup
Requires Python 3.9 or newer. No dependencies or internet connection required.

Create `faq.txt` with questions and answers:

Q: How do I reset my password?
A: Click Forgot Password on the login page.

Q: What are your opening hours?
A: We are open from 9 AM to 5 PM.

Each entry starts with `Q:` and its answer starts with `A:`. Additional nonempty lines after `A:` continue the answer.

## Usage
Ask a question:

python faqbot.py faq.txt 'How can I reset my password?'

Run the built-in tests:

python faqbot.py --test

If no sufficiently similar question is found, the bot reports that it couldn't find a matching answer.
