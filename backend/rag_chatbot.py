"""
rag_chatbot.py — Local RAG chatbot over UPI transactions ("Ask My Statement").

WHAT "RAG" MEANS HERE:
Retrieval-Augmented Generation = instead of an AI model guessing an answer
from what it was trained on, we first RETRIEVE the actual relevant rows
from OUR data, then use those rows to answer. Here the "retrieval" part is
a local vector database (ChromaDB); we stop after retrieval + a plain
calculation, so there's no external LLM call and no API key needed —
everything runs on your machine.

WHAT AN "EMBEDDING" IS:
Every transaction description gets converted into a list of numbers (a
vector) that captures its *meaning*, not just its exact words. Two pieces
of text about similar things end up with similar vectors, even if they
don't share the same words. That's what lets a question like "food
delivery" match a document that says "Swiggy" without us hard-coding that
link everywhere.

PIPELINE:
    raw UPI string --(normalize.py)--> clean (merchant, category)
                   --(this file)-----> stored as a document + metadata in ChromaDB
    user question  --(this file)-----> answered by filtering/searching that data
"""

import re
import chromadb
from chromadb.utils import embedding_functions

from normalize import clean_transaction, MERCHANT_INFO

# ---------------------------------------------------------------------------
# SETUP
# ---------------------------------------------------------------------------

# This model turns text into embeddings. It runs locally on your machine —
# no API key, no per-request cost. The very first time you run this file it
# needs internet to download the model (a few hundred MB); after that it
# works fully offline.
_embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# chromadb.Client() with no arguments = an IN-MEMORY database (exactly what
# the assignment asks for). If you want it to survive between runs instead,
# swap this one line for: chromadb.PersistentClient(path="./chroma_db")
_client = chromadb.Client()
collection = _client.get_or_create_collection(
    name="upi_transactions", embedding_function=_embedder
)

CATEGORIES = {cat for _, cat in MERCHANT_INFO.values()} | {"other"}
MONTH_NAMES = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
}
# Words that signal "this question wants a total", not "browse some examples"
AGGREGATION_HINTS = ["how much", "total", "sum", "spent", "spend", "count"]


# ---------------------------------------------------------------------------
# INDEXING — Task: "index the cleaned transaction rows"
# ---------------------------------------------------------------------------

def index_transactions(rows):
    """
    rows: list of dicts, each with keys: id, date ("YYYY-MM-DD"), amount,
          raw_upi_string.
    Cleans each raw string with normalize.py, turns it into one sentence of
    text (what gets embedded) plus a metadata dict (what we can filter on
    exactly, no embedding needed), and adds both to the collection.
    """
    docs, metadatas, ids = [], [], []
    for r in rows:
        cleaned = clean_transaction(r["raw_upi_string"])
        merchant, category = cleaned["merchant"], cleaned["category"]

        # This sentence is what gets embedded — it's what semantic search
        # actually compares against.
        docs.append(
            f"On {r['date']}, paid Rs.{r['amount']} to {merchant} ({category}) via UPI."
        )
        # Metadata is stored as exact, filterable fields alongside the
        # embedding — no similarity search involved when we filter on these.
        metadatas.append({
            "merchant": merchant,
            "category": category,
            "amount": float(r["amount"]),
            "month": r["date"][:7],   # "2026-03"
            "date": r["date"],
        })
        ids.append(str(r["id"]))

    collection.add(documents=docs, metadatas=metadatas, ids=ids)


# ---------------------------------------------------------------------------
# QUERY UNDERSTANDING — Task: "handle semantic query matching"
# ---------------------------------------------------------------------------

def _mentions(term: str, text: str) -> bool:
    """True if `term` appears in `text` as a whole word/phrase, not buried
    inside another word (so "Ola" doesn't match inside "total")."""
    return re.search(rf"\b{re.escape(term.lower())}\b", text) is not None


def parse_query(query: str, default_year: str = "2026") -> dict:
    """Pull a specific merchant, or else a category, plus an optional month,
    out of a plain-English question."""
    q = query.lower()

    merchant = None
    for display_name, _cat in MERCHANT_INFO.values():
        if _mentions(display_name, q):
            merchant = display_name
            break

    category = None
    if not merchant:  # only look for a category phrase if no merchant was named directly
        category = next((c for c in CATEGORIES if _mentions(c, q)), None)

    month = None
    for name, num in MONTH_NAMES.items():
        if _mentions(name, q):
            month = f"{default_year}-{num}"
            break

    return {"merchant": merchant, "category": category, "month": month}


def _build_where(filters: dict):
    """Turn the parsed filters into a ChromaDB `where` clause."""
    conditions = []
    if filters["month"]:
        conditions.append({"month": filters["month"]})
    if filters["merchant"]:
        conditions.append({"merchant": filters["merchant"]})
    elif filters["category"]:
        conditions.append({"category": filters["category"]})

    if len(conditions) == 2:
        return {"$and": conditions}
    if conditions:
        return conditions[0]
    return None


# ---------------------------------------------------------------------------
# ANSWERING
# ---------------------------------------------------------------------------

def answer_query(query: str, top_k: int = 10) -> dict:
    """
    IMPORTANT DESIGN POINT:
    A pure semantic search (`collection.query`) only returns the top_k most
    SIMILAR rows — great for browsing, but WRONG for a "how much did I
    spend" question, because it can silently drop matching rows past top_k
    and undercount the total. So: whenever we can build an exact filter
    (merchant/category/month), we use `collection.get(where=...)`, which
    returns every matching row with no similarity cutoff. We only fall back
    to true semantic search when the question doesn't give us a clean
    filter to anchor on.
    """
    filters = parse_query(query)
    where = _build_where(filters)
    is_aggregation = any(h in query.lower() for h in AGGREGATION_HINTS)

    if is_aggregation or where:
        got = collection.get(where=where) if where else collection.get()
        rows = got["metadatas"]
    else:
        res = collection.query(query_texts=[query], n_results=top_k)
        rows = res["metadatas"][0]

    total = sum(r["amount"] for r in rows)
    return {
        "total": round(total, 2),
        "count": len(rows),
        "transactions": sorted(rows, key=lambda r: r["date"]),
    }


# ---------------------------------------------------------------------------
# DEMO / CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sample_data import SAMPLE_ROWS

    index_transactions(SAMPLE_ROWS)
    print(f"Indexed {len(SAMPLE_ROWS)} transactions.\n")

    demo_questions = [
        "How much did I spend on food delivery in March?",
        "What did I spend on Netflix?",
        "Total spend on groceries",
    ]
    for q in demo_questions:
        result = answer_query(q)
        print(f"Q: {q}")
        print(f"A: Rs.{result['total']} across {result['count']} transaction(s)\n")

    print("Ask your own question (or type 'quit'):")
    while True:
        q = input("> ").strip()
        if q.lower() in ("quit", "exit"):
            break
        if not q:
            continue
        result = answer_query(q)
        print(f"Rs.{result['total']} across {result['count']} transaction(s)")
        for t in result["transactions"][:5]:
            print(f"   {t['date']}  {t['merchant']:<20} Rs.{t['amount']}")
