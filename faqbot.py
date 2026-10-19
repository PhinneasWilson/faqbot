#!/usr/bin/env python3
"""Offline FAQ chatbot using pure-Python TF-IDF cosine similarity."""

import math
import re
import sys
import unittest
from collections import Counter


def tokenize(text):
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


def load_faq(path):
    entries = []
    question = None
    answer = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            if line.startswith("Q:"):
                if question is not None:
                    entries.append((question, " ".join(answer)))
                question, answer = line[2:].strip(), []
            elif line.startswith("A:") and question is not None:
                answer.append(line[2:].strip())
            elif question is not None:
                answer.append(line)
    if question is not None:
        entries.append((question, " ".join(answer)))
    return entries


def vectorize(text, idf):
    counts = Counter(tokenize(text))
    total = sum(counts.values()) or 1
    return {word: count / total * idf.get(word, 0) for word, count in counts.items()}


def retrieve(question, entries, threshold=0.05):
    if not entries or not tokenize(question):
        return None
    documents = [tokenize(q) for q, _ in entries]
    df = Counter(word for doc in documents for word in set(doc))
    idf = {word: math.log((1 + len(documents)) / (1 + count)) + 1 for word, count in df.items()}
    vectors = [vectorize(q, idf) for q, _ in entries]
    query = vectorize(question, idf)

    def cosine(a, b):
        dot = sum(value * b.get(word, 0) for word, value in a.items())
        norm_a = math.sqrt(sum(value ** 2 for value in a.values()))
        norm_b = math.sqrt(sum(value ** 2 for value in b.values()))
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0

    scores = [cosine(query, vector) for vector in vectors]
    best = max(range(len(scores)), key=scores.__getitem__)
    return entries[best][1] if scores[best] >= threshold else None


def main():
    if len(sys.argv) == 2 and sys.argv[1] == "--test":
        unittest.main(argv=[sys.argv[0]])
        return
    if len(sys.argv) != 3:
        print("Usage: python faqbot.py faq.txt 'your question'")
        sys.exit(2)
    try:
        entries = load_faq(sys.argv[1])
    except OSError as error:
        sys.exit(f"Cannot read FAQ file: {error}")
    answer = retrieve(sys.argv[2], entries)
    print(answer if answer else "Sorry, I couldn't find a matching answer.")


class FAQBotTests(unittest.TestCase):
    def setUp(self):
        self.entries = [
            ("How do I reset my password?", "Use password reset."),
            ("What are your opening hours?", "We open at nine."),
        ]

    def test_matching_question(self):
        self.assertEqual(
            retrieve("How can I reset my password?", self.entries),
            "Use password reset.",
        )

    def test_unrelated_question(self):
        self.assertIsNone(retrieve("purple elephants", self.entries))

    def test_empty_faq(self):
        self.assertIsNone(retrieve("hello", []))

    def test_empty_question(self):
        self.assertIsNone(retrieve("", self.entries))


if __name__ == "__main__":
    main()
