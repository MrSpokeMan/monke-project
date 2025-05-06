QA_GENERATION_PROMPT = """
Your task is to write a factoid question and an answer given a context.
Your factoid question should be answerable with a specific, concise piece of factual information from the context.
Your factoid question should be formulated in the same style as questions users could ask in a search engine.
This means that your factoid question MUST NOT mention something like "according to the passage" or "context".

Provide your answer as follows:

Output:::
Factoid question: (your factoid question)
Answer: (your answer to the factoid question)

Now here is the context.

Context: {context}\n
Output:::
"""

QA_CRITIQUE_GROUNDEDNESS = """
You will be given a context and a question.
Your task is to provide a 'total rating' scoring how well one can answer the given question unambiguously with the given context.
Give your answer on a scale of 1 to 5, where 1 means that the question is not answerable at all given the context, and 5 means that the question is clearly and unambiguously answerable with the context.

Provide your answer as follows:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here are the question and context.

Question: {question}\n
Context: {context}\n
Answer:::
"""

QA_CRITIQUE_RELEVANCE = """
You will be given a question.
Your task is to provide a 'total rating' representing how useful this question is for machine learning applications that process or generate outputs based on legal and regulatory documents from EUR-Lex, particularly in industrial or technical domains (e.g., safety compliance, product standards, environmental regulations).

Rate on a scale from 1 to 5:
- 1 means the question is not useful at all in helping extract, reason about, or generate relevant legal or regulatory information.
- 5 means the question is highly useful for legal reasoning, document analysis, or compliance-related NLP tasks involving EU law.

Questions that refer to specific directives, legal requirements, obligations, definitions, or rights are generally more useful. Vague, overly general, or off-topic questions should receive a lower score.

For example:
- "What obligations does the manufacturer have under Regulation (EU) 2019/1020?" → likely **5**
- "What do they mean?" → likely **1**, due to lack of specificity or legal relevance

Provide your evaluation in the following format:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here is the question.

Question: {question}\n
Answer:::
"""

QA_CRITIQUE_STANDALONE = """
You will be given a question.
Your task is to provide a 'total rating' representing how context-independent this question is, specifically in the domain of European legal or regulatory texts (e.g., directives, regulations, or case law from EUR-Lex), especially those relevant to industrial contexts.

Rate on a scale from 1 to 5:
- 1 means the question heavily depends on external context or document references to be understood.
- 5 means the question is fully self-contained and understandable without requiring additional context.

Legal or technical terms (e.g., 'CE marking', 'harmonised standard', 'REACH regulation') may appear in well-formed standalone questions, as long as the meaning or intent of the question is clear to an operator or developer familiar with EU legal language and access to documentation.

For example, a question like "What are the essential requirements under Directive 2014/35/EU?" should receive a **5**, while "What are they according to the document?" should receive a **1** due to its vague dependency on prior context.

Provide your evaluation in the following format:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here is the question.

Question: {question}\n
Answer:::
"""

