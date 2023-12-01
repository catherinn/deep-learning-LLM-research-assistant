## Open AI API Pricing

The price is calculated for each batch of 1,000 tokens.

100 tokens = 75 words

This applies to both input and output words.

Estimated costs for 100 pages of text:
100 pages of text = 25,000 words
A summary reduces the text size to 20% of the original: 20 pages of text = 5,000 words
Total word count = 30,000 words
Total token count = 39,000 tokens

Cost with the ADA model: $15.6

Model: ADA $0.0004 per 1,000 tokens
Model: Babbage $0.0005 per 1,000 tokens
Model: Curie $0.002 per 1,000 tokens
Model: Davinci $0.02 per 1,000 tokens -> best model

## Using

Istall the dependecies:

'''python
pip install -r tequirements.txt
'''

cf. code: open_api.py