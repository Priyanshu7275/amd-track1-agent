"""
Hand-written seed examples — the foundation of the training corpus.

180 examples across the 8 Track 1 categories, written by hand to establish
the exact output *shape* expected for each category. These were then expanded
to 5,557 examples via synthetic generation (see generate_data.py).

The shape matters more than the content. A sentiment example teaches the model
to answer "mixed" — not "The sentiment here is mixed, because while the
reviewer praises the screen, they criticise the battery life."

A representative subset is shown below; the full file follows the same pattern
with the distribution listed at the bottom.
"""

SEED_EXAMPLES = [
    # ---- FACTUAL (30) -------------------------------------------------------
    # Terse, direct answers. No "The answer is..." preamble.
    {"category": "factual", "prompt": "What is the capital of France?",
     "answer": "Paris is the capital of France, located on the River Seine."},
    {"category": "factual", "prompt": "Who wrote Romeo and Juliet?",
     "answer": "William Shakespeare wrote Romeo and Juliet."},
    {"category": "factual", "prompt": "In what year did World War 2 end?",
     "answer": "World War 2 ended in 1945."},
    {"category": "factual", "prompt": "What element has atomic number 1?",
     "answer": "Hydrogen has atomic number 1."},

    # ---- MATH (30) ----------------------------------------------------------
    # Bare numbers only. This is what stops the model from writing out
    # its (frequently wrong) arithmetic reasoning.
    {"category": "math", "prompt": "What is 25% of 200?",
     "answer": "50"},
    {"category": "math", "prompt": "A store has 240 items. It sells 15% on Monday and 60 more on Tuesday. How many items remain?",
     "answer": "144"},
    {"category": "math", "prompt": "If a train travels at 60 km/h for 2.5 hours, how far does it go?",
     "answer": "150"},
    {"category": "math", "prompt": "If x = 5, what is 3x + 7?",
     "answer": "22"},

    # ---- SENTIMENT (25) -----------------------------------------------------
    # Exactly one word. Includes 'mixed' cases with an explicit contrast marker,
    # which the base model consistently mislabelled as positive or negative.
    {"category": "sentiment", "prompt": "Classify the sentiment: 'I absolutely love this product, it works perfectly!'",
     "answer": "positive"},
    {"category": "sentiment", "prompt": "Classify the sentiment: 'The battery life is great, but the screen scratches too easily.'",
     "answer": "mixed"},
    {"category": "sentiment", "prompt": "Classify the sentiment: 'The item arrived on time. Nothing special.'",
     "answer": "neutral"},
    {"category": "sentiment", "prompt": "Classify the sentiment: 'This is the worst purchase I have ever made.'",
     "answer": "negative"},

    # ---- SUMMARIZATION (20) -------------------------------------------------
    # One sentence, no "This passage describes..." scaffolding.
    {"category": "summarization",
     "prompt": "Summarize in one sentence: The Amazon rainforest, spanning nine countries in South America, produces about 20 percent of Earth's oxygen and hosts millions of species of plants and animals.",
     "answer": "The Amazon rainforest is a vast South American ecosystem producing much of Earth's oxygen and harboring immense biodiversity."},
    {"category": "summarization",
     "prompt": "Summarize in one sentence: Photosynthesis is the process by which plants and some other organisms use sunlight, water, and carbon dioxide to produce oxygen and glucose.",
     "answer": "Photosynthesis is how plants convert sunlight, water, and CO2 into oxygen and glucose."},

    # ---- NER (20) -----------------------------------------------------------
    # Strict "Entity (TYPE)" format, comma separated.
    {"category": "ner",
     "prompt": "Extract all named entities and their types from: Sundar Pichai became CEO of Google in Mountain View in August 2015.",
     "answer": "Sundar Pichai (PERSON), Google (ORG), Mountain View (GPE), August 2015 (DATE)"},
    {"category": "ner",
     "prompt": "Extract all named entities and their types from: Marie Curie was born in Warsaw in 1867 and later worked at the University of Paris.",
     "answer": "Marie Curie (PERSON), Warsaw (GPE), 1867 (DATE), University of Paris (ORG)"},

    # ---- CODE_DEBUG (20) ----------------------------------------------------
    # Corrected code only, no explanation of what was wrong.
    {"category": "code_debug",
     "prompt": "Fix this function that should return the maximum of two numbers: def max_two(a, b): return a",
     "answer": "def max_two(a, b):\n    return a if a > b else b"},
    {"category": "code_debug",
     "prompt": "This factorial function is buggy: def factorial(n): return n * factorial(n)",
     "answer": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)"},

    # ---- LOGIC (15) ---------------------------------------------------------
    # Just the name. No working shown.
    {"category": "logic",
     "prompt": "Three friends, Sam, Jo, and Lee, each own a different pet: cat, dog, bird. Sam does not own the bird. Jo owns the dog. Who owns the cat?",
     "answer": "Sam"},
    {"category": "logic",
     "prompt": "Alice, Bob, and Carol each drive a different color car: red, blue, green. Alice does not drive the red car. Bob drives the green car. Who drives the blue car?",
     "answer": "Alice"},

    # ---- CODE_GEN (20) ------------------------------------------------------
    # Code only, no markdown fences, no commentary.
    {"category": "code_gen",
     "prompt": "Write a Python function that reverses a string.",
     "answer": "def reverse_string(s):\n    return s[::-1]"},
    {"category": "code_gen",
     "prompt": "Write a Python function that returns the second-largest number in a list, handling duplicates correctly.",
     "answer": "def second_largest(numbers):\n    unique = list(set(numbers))\n    if len(unique) < 2:\n        return None\n    unique.sort(reverse=True)\n    return unique[1]"},
]

# Full distribution across the 180 hand-written examples:
#
#   factual        30      code_debug     20
#   math           30      code_gen       20
#   sentiment      25      ner            20
#   summarization  20      logic          15

if __name__ == "__main__":
    from collections import Counter
    print(f"Seed examples: {len(SEED_EXAMPLES)}")
    for category, count in sorted(Counter(e["category"] for e in SEED_EXAMPLES).items()):
        print(f"  {category:<16} {count}")
