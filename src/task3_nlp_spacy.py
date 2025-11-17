# -*- coding: utf-8 -*-
#
# Task 3: NLP with spaCy
# Objective: Perform Named Entity Recognition (NER) and rule-based Sentiment Analysis on user reviews.
# Framework: spaCy (Natural Language Processing)

import spacy
from spacy.lang.en.stop_words import STOP_WORDS

# Note: The model 'en_core_web_sm' must be downloaded separately 
# (Ex: python -m spacy download en_core_web_sm)

print("--- Task 3: NLP - NER and Rule-based Sentiment ---")

# 1. Sample Text Data (Reviews)

amazon_reviews = [
    "The new 'ChronoWatch X1' is absolutely fantastic! Battery life is great, and the 'Ogetec' brand delivered quickly.",
    "I was disappointed with the 'ZenBook 14'. It overheats constantly. I regret buying this laptop.",
    "This book, 'The AI Engineer', is highly informative and a must-read for any developer. Five stars!",
    "The customer service from 'EcoGoods' was terrible. My package arrived late and damaged."
]

# 2. Load spaCy Model 
try:
    # Load the English model 
    nlp = spacy.load("en_core_web_sm")
    print("\nspaCy model 'en_core_web_sm' loaded successfully.")
except OSError:
    print("\n[ERROR] spaCy model 'en_core_web_sm' not found.")
    print("Please run: python -m spacy download en_core_web_sm")
    exit()

# 3. Named Entity Recognition (NER) 
print("\n3. Named Entity Recognition (NER) Results:")

def perform_ner(text):
    """
    Applies the spaCy model to extract named entities.
    """
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

for i, review in enumerate(amazon_reviews):
    entities = perform_ner(review)
    
    # Filter for entities relevant to products/brands 
    product_brands = [(text, label) for text, label in entities if label in ('ORG', 'PRODUCT', 'WORK_OF_ART')]
    
    print(f"\nReview {i+1}: '{review[:50]}...'")
    if product_brands:
        print(f"  Extracted Entities (Product/Brand): {product_brands}")
    else:
        print("  No relevant entities found.")


# 4. Rule-Based Sentiment Analysis 
# This is a simple, rule-based approach using keywords, as required by the task.

positive_words = {"fantastic", "great", "highly informative", "must-read", "five stars", "excellent", "love"}
negative_words = {"disappointed", "overheats", "regret", "terrible", "late", "damaged", "awful"}

def analyze_sentiment(text):
    """
    Performs basic sentiment analysis by counting positive and negative keywords.
    """
    doc = nlp(text.lower())
    
    # Tokenization and removing stop words 
    tokens = [token.text for token in doc if token.text not in STOP_WORDS and token.is_alpha]
    
    pos_count = sum(1 for token in tokens if token in positive_words)
    neg_count = sum(1 for token in tokens if token in negative_words)
    
    if pos_count > neg_count:
        return "Positive"
    elif neg_count > pos_count:
        return "Negative"
    else:
        return "Neutral/Mixed"

print("\n4. Rule-Based Sentiment Analysis Results:")
for i, review in enumerate(amazon_reviews):
    sentiment = analyze_sentiment(review)
    print(f"Review {i+1}: Sentiment: {sentiment}")

print("\nTask 3 completed successfully. NER and Sentiment analysis performed.")