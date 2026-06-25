# Jargon Reference

Plain-English definitions for terms encountered across this project.
Grouped by topic. Add new entries as they come up.

---

## Vectors & Embeddings

**Embedding**
The process of converting a piece of text (a word, sentence, or document) into a list of
numbers that captures its meaning. An embedding model reads the text and outputs a fixed-size
array of floats — typically 768 or 1536 numbers. Texts with similar meanings produce similar
arrays. This is what makes semantic search possible.

**Vector**
The list of numbers produced by an embedding. "Vector" and "embedding" are often used
interchangeably in this context. Technically, a vector is just the mathematical object
(an ordered list of numbers); an embedding is what that vector represents (encoded meaning).

**Embedding model**
The model that performs the text-to-vector conversion. In this project: `nomic-embed-text`
running in Ollama. It is separate from the LLM — it does not generate text, only encodes
meaning into numbers. Faster and cheaper than a full LLM.

**Cosine similarity**
A measure of how similar two vectors are, based purely on the *angle* between them — not
their size. Score ranges from 0.0 (pointing in completely different directions, unrelated)
to 1.0 (pointing in exactly the same direction, identical meaning). The standard metric for
comparing text embeddings because it ignores length and focuses on intent.

**L2 normalization**
Scaling a vector so its total length becomes exactly 1.0, while preserving its direction.
After normalization, two vectors can be compared by inner product and the result equals
cosine similarity. This is the standard preprocessing step before loading vectors into FAISS.

**Dimensionality / Dimension (D)**
The number of values in a single embedding vector. `nomic-embed-text` produces 768-dimensional
vectors. Higher dimensions can capture more nuance but require more memory and compute.

---

## Retrieval & RAG

**RAG (Retrieval-Augmented Generation)**
A pattern where the LLM is given relevant documents fetched from a knowledge base before
generating a response, rather than relying solely on its training memory. The retrieval step
provides grounding; the generation step uses that grounding to produce an answer.
"Augmented" refers to augmenting the prompt with retrieved content.

**Knowledge base**
The collection of documents (or past records) the retriever searches over. In this project:
`data/knowledge_base.json` — 30 past resolved support tickets. In production this could be
a database of product documentation, past resolutions, or any reference material.

**Retriever**
The component responsible for searching the knowledge base and returning the most relevant
documents for a given query. Takes a text query, embeds it, compares against the stored
embeddings, and returns the top-K closest matches.

**Top-K**
The number of results the retriever returns. K=3 means "give me the 3 most similar past
resolutions." Higher K gives the LLM more context but risks including irrelevant content
that confuses the model. There is always a tradeoff between coverage and noise.

**Semantic search**
Finding documents by meaning rather than by matching exact words. "App keeps crashing" and
"application has a recurrent stability problem" return the same results in semantic search
because their embeddings point in the same direction, even though they share no keywords.

**Keyword search (BM25)**
The traditional approach: find documents that contain the same words as the query. Fast and
explainable, but fails on paraphrase, synonyms, and domain jargon. Semantic search is almost
always better for natural language queries.

**Grounding / Groundedness**
A response is "grounded" when every claim it makes can be traced back to the provided
context. A grounded reply about a bug fix references the specific cause found in the
retrieved resolution, rather than inventing a plausible-sounding explanation.

**Hallucination**
When a model generates confident-sounding content that is factually incorrect or completely
fabricated. RAG reduces hallucination by giving the model real content to work from, but
does not eliminate it — the model can still ignore or misinterpret the retrieved context.

**Context window**
The maximum amount of text (measured in tokens) an LLM can read in a single call —
everything: system prompt, retrieved context, conversation history, and the user's message.
If the retrieved context is too large, it fills the window and leaves no room for the rest.
This is why K and chunk size matter.

**Chunking**
Splitting long documents into smaller pieces before embedding and indexing them. A 10-page
PDF embedded as one vector loses detail; split into paragraphs, each chunk captures a
specific idea and retrieval becomes more precise. Not needed for this project's short
resolutions, but critical at production scale.

---

## FAISS

**FAISS**
Facebook AI Similarity Search — a library for fast nearest-neighbor search over large
collections of vectors. You load all your embeddings into a FAISS index at startup; at
query time it finds the closest vectors in milliseconds, even across millions of entries.

**IndexFlatIP**
A FAISS index type that ranks vectors by inner product (IP = inner product). When vectors
are L2-normalized first, inner product equals cosine similarity, so this becomes a cosine
similarity index. Returns the highest-scoring (most similar) results first.

**IndexFlatL2**
A FAISS index type that ranks vectors by Euclidean (straight-line) distance. Returns the
lowest-distance (most similar) results first. Works for cosine similarity after normalization
too, but the score interpretation is inverted — lower is better, which is less intuitive.

**Inner product**
A mathematical operation on two vectors that multiplies their corresponding values and sums
the results. On unit vectors (length = 1.0 after normalization) this equals cosine similarity.
On un-normalized vectors it is affected by both direction and magnitude.

---

## DSPy

**Signature**
A class that declares what goes into an LLM call and what must come out, without writing
prompt text. DSPy reads the field names, types, and `desc=` strings and constructs the
prompt automatically. Think of it as a typed function contract for an LLM.

**Module**
A composable unit that wraps one or more predictors. Mirrors PyTorch's `nn.Module` pattern:
`__init__` declares the predictors; `forward()` wires the data flow. Modules can be nested —
a pipeline module contains a classify module and a reply module.

**Predictor**
The object that actually calls the LLM. `dspy.Predict` gives a direct answer;
`dspy.ChainOfThought` adds a hidden reasoning step before the answer. Both are initialized
with a Signature that defines their input/output contract.

**ChainOfThought (CoT)**
A predictor that forces the model to produce a reasoning trace before committing to the
final answer. The reasoning is a hidden field — the model fills it in, and it influences
the answer, but you define only the public output fields in the Signature.

**Compilation / compile()**
The DSPy optimizer's process of improving a module's prompt by finding better instructions
and demonstrations. `compile()` runs the module over a training set, measures quality with
a metric function, and injects the best-performing examples into the prompt. The result is
a new module with the same Signature but an improved prompt baked in.

**Demonstrations**
Concrete examples of correct input-output pairs injected into the prompt by the optimizer.
A demonstration shows the model "when you see a ticket like this, produce an answer like
this." BootstrapFewShot finds these by running the module and keeping the examples it got
right.

**BootstrapFewShot**
The simplest DSPy optimizer. Runs the module on the training set, keeps examples where the
metric passed, and injects them as demonstrations. Fast. Good starting point.

**MIPROv2**
The current state-of-the-art DSPy optimizer. Jointly optimizes both the task instruction
(the Signature docstring equivalent) and the demonstrations. Takes more LLM calls than
Bootstrap but usually achieves higher final accuracy.

**Metric function**
A Python function with signature `(example, pred, trace=None) -> float` that scores one
prediction against the ground truth. Used by `dspy.Evaluate` to measure module accuracy and
by the optimizer to decide which examples become demonstrations. Returns a float in [0, 1].

**Reward function**
A Python function with signature `(inputs: dict, pred: Prediction) -> float` used by
`dspy.Refine`. Scores a live prediction — not a dataset example. Same float-in-[0,1]
convention as metric functions but receives the actual call inputs, not a dataset row.

**dspy.Refine**
A wrapper that runs a module up to N times, scores each output with a reward function,
and returns the best result. If the threshold is not met, it generates feedback via an
internal LLM call and injects it as a hint into the next attempt. Replaces the deprecated
`dspy.Assert` / `dspy.Suggest` from DSPy 2.x.

**inspect_history()**
A DSPy debugging function that prints the exact prompt(s) sent to the LLM in the most
recent call(s). `dspy.inspect_history(n=2)` shows the last 2 calls. Essential for
understanding what DSPy actually constructed from your Signature.

**Jaccard similarity**
A metric for comparing two sets: `|intersection| / |union|`. Used as the classification
metric in M17–M20. For single-label classification it degenerates to exact match (1.0 or
0.0). For multi-label it gives partial credit when some labels overlap.

---

## LLM Fundamentals

**LLM (Large Language Model)**
A neural network trained on vast amounts of text to predict the next token. At inference
time, you give it a prompt and it generates a continuation. The "large" refers to the
number of parameters (weights), which typically ranges from a few billion to hundreds of
billions.

**Token**
The unit an LLM reads and writes. Not exactly a word — a token is typically 3–4 characters.
"unbelievable" might be 3 tokens: "un", "believ", "able". Token counts determine both the
cost (for paid APIs) and the context window limit.

**Temperature**
A parameter that controls how random the model's output is. Temperature=0 makes the model
deterministic (always picks the most likely next token). Temperature=1.0 introduces
variation — the model occasionally picks less likely tokens, producing more diverse outputs.
`dspy.Refine` uses temperature=1.0 across retries specifically to get different outputs.

**Inference**
Running a trained model to produce an output. As opposed to training, which updates the
model's weights. In production you only do inference — training is done once (or rarely).

**Context window**
See *Context window* under Retrieval & RAG above.

**System message**
The first message in a chat prompt, written by the developer, that sets the model's role
and behavior before the user's message arrives. DSPy constructs this automatically from
the Signature.

**Zero-shot**
Running a model on a task with no examples in the prompt — just the instruction. The model
relies entirely on its training. Often surprisingly capable; always worth trying before
adding complexity.

**Few-shot**
Providing a small number of examples in the prompt to show the model what correct outputs
look like. DSPy's optimizers automate the selection of these examples.

---

## Evaluation

**Golden dataset**
A fixed, curated set of input-output pairs used as the permanent benchmark for a system.
Never updated to make tests pass — only grows when new coverage is genuinely needed. The
yardstick against which all prompt changes are measured.

**LLM-as-judge**
Using a (usually stronger) LLM to evaluate the output of another LLM against a rubric.
Solves the phrasing problem of deterministic string matching — the judge understands that
"I will look into this" and "we will investigate immediately" are equivalent answers.
Introduces its own biases (verbosity bias, position bias, self-enhancement bias).

**RAG Triad**
Three evaluation metrics for RAG pipelines, each targeting a different component:
- *Context Relevance*: did retrieval fetch useful content?
- *Faithfulness*: is the answer grounded in the retrieved content?
- *Answer Relevance*: does the answer address the original question?
Popularized by the RAGAS and TruLens frameworks.

**Verbosity bias**
The tendency of LLM judges to score longer answers higher regardless of actual quality.
Mitigation: instruct the judge explicitly to score conciseness and to penalize padding.

**Position bias**
When an LLM judge compares two answers side-by-side, it tends to favor whichever appears
first in the prompt. Mitigation: run the evaluation twice with the order swapped and
average the scores.

---

## Observability

**Span**
A single unit of work in a traced execution. In a multi-step pipeline, each LLM call,
retrieval call, and preprocessing step is its own span. Spans have a start time, end time,
inputs, outputs, and a parent span that ties them into a tree. The full tree from user
request to final response is called a trace.

**Trace**
The complete record of everything that happened during one pipeline execution, structured
as a tree of spans. A trace answers: "given this input, what did every step do, in what
order, and how long did each take?" The foundation of platforms like Langfuse, Arize
Phoenix, and Braintrust.

**Semantic drift**
When a model's outputs gradually change in quality or style over time without any explicit
change to the prompt or model. Can happen when the underlying model is updated by the
provider, when the input distribution shifts (users start asking different questions), or
when retrieved context changes. Only detectable through continuous production monitoring.
