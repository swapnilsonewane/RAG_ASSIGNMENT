from prometheus_client import Counter, Histogram, REGISTRY

# ==================================================
# METRIC SAFE
# ==================================================

def _get_or_create_counter(name: str, documentation: str):
    try:
        return Counter(name, documentation)
    except ValueError:
        return REGISTRY._names_to_collectors[name]


def _get_or_create_histogram(
    name: str,
    documentation: str,
    buckets=None,
):
    try:
        return Histogram(
            name,
            documentation,
            buckets=buckets,
        )
    except ValueError:
        return REGISTRY._names_to_collectors[name]

# ==================================================
# QA METRICS
# ==================================================

REQ_TOTAL = _get_or_create_counter(
    "legal_qa_requests_total",
    "Total QA requests",
)

REQ_FAILURE = _get_or_create_counter(
    "legal_qa_failures_total",
    "QA failures",
)

REQ_CACHE_HIT = _get_or_create_counter(
    "legal_qa_cache_hit_total",
    "QA cache hits",
)

REQ_CACHE_MISS = _get_or_create_counter(
    "legal_qa_cache_miss_total",
    "QA cache misses",
)

NODE_LATENCY_ANSWER = _get_or_create_histogram(
    "qa_answer_latency_seconds",
    "Answer node latency",
    buckets=(0.25, 0.5, 1, 2, 5, 10),
)

NODE_LATENCY_CRITIC = _get_or_create_histogram(
    "qa_critic_latency_seconds",
    "Critic node latency",
    buckets=(0.25, 0.5, 1, 2, 5),
)
