import os
import time
import json
import hashlib
import re
from typing import TypedDict, Literal, List, Optional

import redis.asyncio as redis
from pydantic import BaseModel, ValidationError

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage

from prometheus_client import Counter, Histogram

from multi_agent_rag.services.retrieval_service import retrieve

# ======================================================
# CONFIG
# ======================================================

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
CACHE_TTL_SECONDS = 3600
MAX_RETRIES = 2

CB_THRESHOLD = 3
CB_RESET_SECONDS = 30

# ======================================================
# METRICS
# ======================================================

REQ_TOTAL = Counter("legal_qa_requests_total", "Total QA requests")
REQ_FAILURE = Counter("legal_qa_failures_total", "Failures")
REQ_CACHE_HIT = Counter("legal_qa_cache_hit_total", "Cache hits")
REQ_CACHE_MISS = Counter("legal_qa_cache_miss_total", "Cache misses")

NODE_LATENCY = {
    "answer": Histogram("qa_answer_latency_seconds", "Answer latency"),
    "critic": Histogram("qa_critic_latency_seconds", "Critic latency"),
}

# ======================================================
# REDIS
# ======================================================

redis_client = redis.from_url(REDIS_URL, decode_responses=True)

# ======================================================
# OUTPUT SCHEMA
# ======================================================

class LegalSupport(BaseModel):
    point: str
    citation: str


class LegalAnswer(BaseModel):
    answer: str
    support: List[LegalSupport]

# ======================================================
# STATE
# ======================================================

class GraphState(TypedDict):
    query: str
    user_id: str
    answer: Optional[LegalAnswer]
    verdict: Literal["approve", "reject", ""]
    feedback: str
    retries: int
    trace: List[str]

# ======================================================
# UTILITY
# ======================================================

def cache_key(user_id: str, query: str) -> str:
    h = hashlib.sha256(query.strip().lower().encode()).hexdigest()
    return f"qa:{user_id}:{h}"


async def get_cached_answer(key: str) -> Optional[LegalAnswer]:
    data = await redis_client.get(key)
    if not data:
        return None
    try:
        return LegalAnswer.model_validate_json(data)
    except ValidationError:
        return None


async def set_cached_answer(key: str, answer: LegalAnswer):
    await redis_client.setex(
        key,
        CACHE_TTL_SECONDS,
        answer.model_dump_json(),
    )


def extract_message_content(result) -> str:
    """Extract text content from agent result (handles AIMessage, dict, or AgentFinish)"""
    if hasattr(result, 'content'):
        return result.content
    
    if isinstance(result, dict):
        if 'messages' in result and len(result['messages']) > 0:
            last_msg = result['messages'][-1]
            if hasattr(last_msg, 'content'):
                return last_msg.content
            return str(last_msg)
        
        if 'output' in result:
            return result['output']
    
    return str(result)


def parse_answer_output(text: str) -> LegalAnswer:
    """Parse the answer agent's text output into structured LegalAnswer"""
    
    if not re.search(r'Legal Support:|Citation:', text, re.IGNORECASE):

        return LegalAnswer(
            answer=text.strip(),
            support=[LegalSupport(
                point="Conversational response",
                citation="N/A - No legal research required"
            )]
        )

    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return LegalAnswer.model_validate(data)
        except (json.JSONDecodeError, ValidationError):
            pass

    answer_match = re.search(r'Answer:\s*(.*?)(?=Legal Support:|$)', text, re.DOTALL | re.IGNORECASE)
    answer_text = answer_match.group(1).strip() if answer_match else text.strip()

    support = []
    support_section = re.search(r'Legal Support:(.*)', text, re.DOTALL | re.IGNORECASE)
    if support_section:
        support_text = support_section.group(1)
        point_pattern = r'-\s*(.*?)\s*-\s*Citation:\s*(.*?)(?=\n\s*-|\Z)'
        for match in re.finditer(point_pattern, support_text, re.DOTALL):
            point = match.group(1).strip()
            citation = match.group(2).strip()
            if point and citation:
                support.append(LegalSupport(point=point, citation=citation))

    if not support:
        citation_matches = re.finditer(r'(?:Source|Citation|Ref):\s*([^\n]+)', text, re.IGNORECASE)
        for match in citation_matches:
            support.append(LegalSupport(
                point="Legal information from retrieved documents",
                citation=match.group(1).strip()
            ))

    if not support:
        support.append(LegalSupport(
            point="No specific legal citations found",
            citation="Retrieved from legal documents"
        ))
    
    return LegalAnswer(answer=answer_text, support=support)


def parse_critic_output(text: str) -> tuple[str, str]:
    """Parse critic output to extract verdict and feedback"""

    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        try:
            data = json.loads(json_match.group())
            verdict = data.get("verdict", "reject")
            feedback = data.get("feedback", "")
            return verdict, feedback
        except json.JSONDecodeError:
            pass
    
    verdict = "reject"
    feedback = ""
    
    if re.search(r'\bapprove\b', text, re.IGNORECASE):
        verdict = "approve"
    
    feedback_match = re.search(r'feedback["\']?\s*:\s*["\']?(.*?)["\']?[,}]', text, re.IGNORECASE)
    if feedback_match:
        feedback = feedback_match.group(1).strip()
    else:
        feedback = text[:200]
    
    return verdict, feedback

# ======================================================
# REDIS CIRCUIT BREAKER
# ======================================================

class RedisCircuitBreaker:
    def __init__(self, name: str):
        self.key = f"cb:{name}"

    async def allow(self) -> bool:
        state = await redis_client.hgetall(self.key)
        opened_at = state.get("opened_at")
        if not opened_at:
            return True
        if time.time() - float(opened_at) > CB_RESET_SECONDS:
            await redis_client.delete(self.key)
            return True
        return False

    async def failure(self):
        pipe = redis_client.pipeline()
        pipe.hincrby(self.key, "failures", 1)
        pipe.hsetnx(self.key, "opened_at", time.time())
        await pipe.execute()

    async def success(self):
        await redis_client.delete(self.key)

answer_cb = RedisCircuitBreaker("answer")
critic_cb = RedisCircuitBreaker("critic")

# ======================================================
# LLM
# ======================================================

def build_llm(max_tokens: int, timeout: int):
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.0,
        max_output_tokens=max_tokens,
        timeout=timeout,
    )

answer_llm = build_llm(1024, 20)
critic_llm = build_llm(512, 15)

# ======================================================
# AGENTS
# ======================================================

ANSWER_PROMPT = """
You are a legal research assistant.

INSTRUCTIONS:
1. If the user query is a greeting (hello, hi, etc.) or general conversation, respond naturally WITHOUT using tools.
2. For legal questions, you MUST:
   a) First call the query_legal_docs tool to retrieve relevant legal documents
   b) Wait for the tool results
   c) Use ONLY the retrieved passages to formulate your answer
   d) Include citations from the retrieved passages

3. Every legal assertion MUST be supported by a citation from the retrieved documents.
4. If retrieved material is insufficient, incomplete, or conflicting, state this explicitly.
5. Do NOT speculate or add information beyond what's in the retrieved passages.
6. If no relevant documents are retrieved, state that clearly.

OUTPUT FORMAT (STRICT) for legal questions:
Answer:
<concise answer based ONLY on retrieved passages>

Legal Support:
- <specific point from your answer>
  - Citation: <exact citation from retrieved passage>
- <another specific point>
  - Citation: <exact citation from retrieved passage>

For greetings/general conversation, respond naturally without the structured format.
"""


CRITIC_PROMPT = """
You are a legal answer reviewer.

TASK:
Review the answer and evaluate it based on the type of query:

For LEGAL QUESTIONS, check for:
1. Hallucinations - information not present in retrieved sources
2. Unsupported legal claims - assertions without citations
3. Incorrect or irrelevant citations
4. Overgeneralization beyond the source material
5. Missing important information from the retrieved passages

For GREETINGS/CASUAL CONVERSATION:
- If the answer is a simple, appropriate greeting or conversational response, APPROVE it
- These don't require citations or legal support

IMPORTANT:
- Be STRICT for legal questions
- Be LENIENT for greetings and casual conversation
- If citations say "N/A - No legal research required", it's likely a greeting - APPROVE it

OUTPUT JSON ONLY:
{
  "verdict": "approve" | "reject",
  "feedback": "<short reason if rejected, empty if approved>"
}
"""


critic_agent = create_agent(
    critic_llm,
    tools=[],
    system_prompt=CRITIC_PROMPT,
)

# ======================================================
# NODES
# ======================================================

async def answer_node(state: GraphState) -> GraphState:
    start = time.perf_counter()

    key = cache_key(state["user_id"], state["query"])
    cached = await get_cached_answer(key)

    if cached:
        REQ_CACHE_HIT.inc()
        return {
            **state,
            "answer": cached,
            "trace": state["trace"] + ["cache_hit"],
        }

    REQ_CACHE_MISS.inc()

    if not await answer_cb.allow():
        raise RuntimeError("Answer breaker open")

    try:

        user_id = state["user_id"]
        
        async def query_legal_docs_with_user(query: str) -> dict:
            """Query legal docs with user_id from state"""
            result = await retrieve(query, user_id)
            return result.model_dump()
        
        tool_with_user = StructuredTool(
            name="query_legal_docs",
            description="Retrieve legal documents for a user query",
            coroutine=query_legal_docs_with_user,
            args_schema=type("QueryInput", (BaseModel,), {
                "__annotations__": {"query": str},
                "query": (str, ...),
            }),
        )

        agent_with_context = create_agent(
            answer_llm,
            tools=[tool_with_user],
            system_prompt=ANSWER_PROMPT,
        )

        result = await agent_with_context.ainvoke(
            {"messages": [HumanMessage(content=state["query"])]},
        )

        content = extract_message_content(result)

        parsed = parse_answer_output(content)

        await answer_cb.success()

        return {
            **state,
            "answer": parsed,
            "trace": state["trace"] + ["answer"],
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        await answer_cb.failure()
        raise

    finally:
        NODE_LATENCY["answer"].observe(time.perf_counter() - start)


async def critic_node(state: GraphState) -> GraphState:
    start = time.perf_counter()

    if not await critic_cb.allow():
        raise RuntimeError("Critic breaker open")

    try:

        answer_json = json.dumps(state["answer"].model_dump(), indent=2)
        
        result = await critic_agent.ainvoke(
            {"messages": [HumanMessage(content=answer_json)]},
        )

        content = extract_message_content(result)
        
        verdict, feedback = parse_critic_output(content)

        if verdict == "approve":
            key = cache_key(state["user_id"], state["query"])
            await set_cached_answer(key, state["answer"])

        await critic_cb.success()

        return {
            **state,
            "verdict": verdict,
            "feedback": feedback,
            "retries": state["retries"] + (1 if verdict == "reject" else 0),
            "trace": state["trace"] + [f"critic:{verdict}"],
        }

    except Exception as e:
        await critic_cb.failure()
        raise

    finally:
        NODE_LATENCY["critic"].observe(time.perf_counter() - start)

# ======================================================
# ROUTING
# ======================================================

def route_after_critic(state: GraphState):
    if state["verdict"] == "approve":
        return END
    if state["retries"] >= MAX_RETRIES:
        return END
    return "answer"

# ======================================================
# GRAPH
# ======================================================

graph = StateGraph(GraphState)
graph.add_node("answer", answer_node)
graph.add_node("critic", critic_node)

graph.set_entry_point("answer")
graph.add_edge("answer", "critic")
graph.add_conditional_edges("critic", route_after_critic)

app_graph = graph.compile()