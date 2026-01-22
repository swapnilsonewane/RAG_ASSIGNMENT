import os
import json
import asyncio
import traceback
from typing import List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from multi_agent_rag.services.retrieval_service import retrieve
from multi_agent_rag.services.graph import app_graph, GraphState


# ======================================================
# EVALUATION METRICS
# ======================================================

class MetricType(str, Enum):
    RETRIEVAL_PRECISION = "retrieval_precision"
    RETRIEVAL_RECALL = "retrieval_recall"
    RETRIEVAL_MRR = "retrieval_mrr"
    ANSWER_FAITHFULNESS = "answer_faithfulness"
    ANSWER_RELEVANCE = "answer_relevance"
    CITATION_ACCURACY = "citation_accuracy"
    HALLUCINATION_RATE = "hallucination_rate"


# ======================================================
# EVALUATION DATA MODELS
# ======================================================

@dataclass
class EvaluationQuery:
    """Single evaluation query with ground truth"""
    query: str
    ground_truth_answer: str
    category: str
    difficulty: str
    expected_citations: List[str]


@dataclass
class RetrievalMetrics:
    """Retrieval evaluation metrics - latency only"""
    total_retrieved: int
    error: Optional[str] = None


@dataclass
class AnswerMetrics:
    """Answer quality metrics"""
    faithfulness_score: float
    relevance_score: float
    citation_accuracy: float
    hallucination_detected: bool
    completeness_score: float
    reasoning: str
    error: Optional[str] = None


@dataclass
class EvaluationResult:
    """Complete evaluation result for one query"""
    query: str
    category: str
    difficulty: str
    retrieval_metrics: Optional[RetrievalMetrics]
    answer_metrics: Optional[AnswerMetrics]
    generated_answer: str
    ground_truth_answer: str
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    timestamp: str
    passed: bool
    failure_reasons: List[str]
    error: Optional[str] = None



# ======================================================
# EVALUATION DATASET
# ======================================================

class EvaluationDataset:
    """Manages evaluation queries and ground truth"""
    
    def __init__(self):
        self.queries: List[EvaluationQuery] = []
    
    def add_query(self, query: EvaluationQuery):
        self.queries.append(query)
    
    def load_from_json(self, filepath: str):
        """Load evaluation dataset from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
            for item in data:
                self.queries.append(EvaluationQuery(**item))
    
    def save_to_json(self, filepath: str):
        """Save dataset to JSON"""
        with open(filepath, 'w') as f:
            json.dump([asdict(q) for q in self.queries], f, indent=2)
    
    def get_by_category(self, category: str) -> List[EvaluationQuery]:
        return [q for q in self.queries if q.category == category]
    
    def get_by_difficulty(self, difficulty: str) -> List[EvaluationQuery]:
        return [q for q in self.queries if q.difficulty == difficulty]


# ======================================================
# SAMPLE EVALUATION DATASET
# ======================================================

def create_sample_dataset() -> EvaluationDataset:
    """Create sample evaluation dataset from the provided legal documents"""
    dataset = EvaluationDataset()
    
    dataset.add_query(EvaluationQuery(
        query="Are pure omissions actionable under Rule 10b-5(b)?",
        ground_truth_answer="No, pure omissions are not actionable under Rule 10b-5(b). The rule covers half-truths, not pure omissions. It requires identifying affirmative statements ('statements made') before determining if other facts are needed to make those statements 'not misleading.'",
        category="securities_fraud",
        difficulty="medium",
        expected_citations=["Macquarie Infrastructure Corp. v. Moab Partners", "Rule 10b-5(b)", "601 U.S. 257"]
    ))
    
    dataset.add_query(EvaluationQuery(
        query="What is the difference between a half-truth and a pure omission in securities law?",
        ground_truth_answer="A pure omission occurs when a speaker says nothing in circumstances that do not give special significance to that silence. Half-truths are representations that state the truth only so far as it goes, while omitting critical qualifying information. The difference is like a child not telling parents he ate a whole cake versus telling them he had dessert.",
        category="securities_fraud",
        difficulty="easy",
        expected_citations=["Universal Health Services", "579 U.S. 176"]
    ))

    dataset.add_query(EvaluationQuery(
        query="Can risk factor disclosures be misleading if they don't disclose that a warned-of risk has already materialized?",
        ground_truth_answer="Yes, risk factor disclosures can be false or misleading when they present risks as hypothetical that have already materialized. Forward-looking statements of risk can imply representations about past or present facts. However, this depends on context and whether a reasonable investor would view the omission as significant.",
        category="risk_disclosure",
        difficulty="hard",
        expected_citations=["Facebook, Inc. v. Amalgamated Bank", "Item 105", "Regulation S-K"]
    ))

    dataset.add_query(EvaluationQuery(
        query="What does Item 303 of Regulation S-K require companies to disclose?",
        ground_truth_answer="Item 303 requires companies to describe any known trends or uncertainties that have had or that are reasonably likely to have a material favorable or unfavorable impact on net sales or revenues or income from continuing operations.",
        category="disclosure_requirements",
        difficulty="easy",
        expected_citations=["Item 303", "17 CFR §229.303(b)(2)(ii)", "Regulation S-K"]
    ))
    
    dataset.add_query(EvaluationQuery(
        query="What is the materiality standard for securities fraud claims?",
        ground_truth_answer="There must be a substantial likelihood that the disclosure of the omitted fact would have been viewed by the reasonable investor as having significantly altered the 'total mix' of information made available. Materiality is assessed from the perspective of a reasonable investor and is context-dependent.",
        category="securities_fraud",
        difficulty="medium",
        expected_citations=["Basic Inc. v. Levinson", "485 U.S. 224", "Matrixx Initiatives"]
    ))
    

    dataset.add_query(EvaluationQuery(
        query="How does Section 10(b) differ from Section 11(a) regarding omissions?",
        ground_truth_answer="Section 11(a) of the Securities Act creates liability for failure to state a material fact 'required to be stated' or 'necessary to make the statements therein not misleading.' Section 10(b) and Rule 10b-5(b) do not contain similar language creating liability for pure omissions. Section 10(b) catches fraud, not all omissions.",
        category="securities_fraud",
        difficulty="hard",
        expected_citations=["15 U.S.C. §77k(a)", "15 U.S.C. §78j(b)", "Ernst & Ernst v. Hochfelder"]
    ))

    dataset.add_query(EvaluationQuery(
        query="In the Facebook case, why were the risk factor statements considered potentially misleading?",
        ground_truth_answer="Facebook's risk factor statements presented the risk of improper user data disclosure as hypothetical, using language like 'could' and 'if,' when Cambridge Analytica had already improperly obtained data from tens of millions of Facebook users. The statements created a false impression that such a breach had not occurred, even though Facebook knew about it.",
        category="risk_disclosure",
        difficulty="medium",
        expected_citations=["In re Alphabet, Inc. Sec. Litig.", "Berson v. Applied Signal Tech."]
    ))

    dataset.add_query(EvaluationQuery(
        query="Does a duty to disclose under Item 303 automatically create liability under Rule 10b-5(b)?",
        ground_truth_answer="No, a duty to disclose does not automatically render silence misleading under Rule 10b-5(b). The failure to disclose information required by Item 303 can support a Rule 10b-5(b) claim only if the omission renders affirmative statements made misleading. Pure omissions are not actionable.",
        category="omission_liability",
        difficulty="hard",
        expected_citations=["Macquarie Infrastructure Corp.", "Basic Inc. v. Levinson", "485 U.S. 224"]
    ))
    
    return dataset

# ======================================================
# RETRIEVAL EVALUATOR
# ======================================================

class RetrievalEvaluator:
    """Evaluates retrieval performance - tracks latency only"""
    
    @staticmethod
    async def evaluate_retrieval(
        query: str,
        user_id: str,
        k: int = 5
    ) -> RetrievalMetrics:
        """Evaluate retrieval - only tracking latency, no quality metrics"""
        try:
            results = await retrieve(query, user_id)
            
            return RetrievalMetrics(
                total_retrieved=len(results.passages),
                error=None
            )
        
        except Exception as e:
            error_msg = f"Retrieval failed: {str(e)}"
            traceback.print_exc()
            
            return RetrievalMetrics(
                total_retrieved=0,
                error=error_msg
            )

# ======================================================
# ANSWER EVALUATOR (LLM-as-Judge)
# ======================================================

class AnswerEvaluator:
    """Evaluates answer quality using LLM-as-judge"""
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.0,
            max_output_tokens=2048,
        )
    
    async def evaluate_answer(
        self,
        query: str,
        generated_answer: str,
        ground_truth: str,
        retrieved_passages: List[str],
        citations: List[str]
    ) -> AnswerMetrics:
        """Evaluate answer using LLM as judge with error handling"""
        
        try:
            evaluation_prompt = f"""
You are an expert evaluator for a legal Q&A system. Evaluate the generated answer across multiple dimensions.

QUERY:
{query}

GROUND TRUTH ANSWER:
{ground_truth}

GENERATED ANSWER:
{generated_answer}

RETRIEVED PASSAGES:
{json.dumps(retrieved_passages[:3], indent=2)}

CITATIONS PROVIDED:
{json.dumps(citations, indent=2)}

Evaluate the answer on these dimensions (score 0.0 to 1.0):

1. FAITHFULNESS: Is the answer faithful to the retrieved passages? Does it contain information not in the sources?
2. RELEVANCE: Does the answer directly address the query?
3. CITATION_ACCURACY: Are the citations accurate and properly support the claims?
4. HALLUCINATION: Does the answer contain fabricated information?
5. COMPLETENESS: Does it cover the key points from the ground truth?

Return ONLY a JSON object with this structure:
{{
  "faithfulness_score": 0.0-1.0,
  "relevance_score": 0.0-1.0,
  "citation_accuracy": 0.0-1.0,
  "hallucination_detected": true/false,
  "completeness_score": 0.0-1.0,
  "reasoning": "Brief explanation of scores"
}}
"""
            
            result = await self.llm.ainvoke([HumanMessage(content=evaluation_prompt)])
            content = result.content.strip()
            
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            data = json.loads(content)
            
            return AnswerMetrics(
                faithfulness_score=data["faithfulness_score"],
                relevance_score=data["relevance_score"],
                citation_accuracy=data["citation_accuracy"],
                hallucination_detected=data["hallucination_detected"],
                completeness_score=data["completeness_score"],
                reasoning=data["reasoning"],
                error=None
            )
        
        except Exception as e:
            error_msg = f"Answer evaluation failed: {str(e)}"
            traceback.print_exc()
            
            return AnswerMetrics(
                faithfulness_score=0.0,
                relevance_score=0.0,
                citation_accuracy=0.0,
                hallucination_detected=True,
                completeness_score=0.0,
                reasoning=error_msg,
                error=error_msg
            )


# ======================================================
# MAIN EVALUATION PIPELINE
# ======================================================

class RAGEvaluationPipeline:
    """Complete RAG evaluation pipeline with separate retrieval and generation evaluation"""
    
    def __init__(self, user_id: str = "eval_user"):
        self.user_id = user_id
        self.retrieval_evaluator = RetrievalEvaluator()
        self.answer_evaluator = AnswerEvaluator()
        self.results: List[EvaluationResult] = []
    
    async def evaluate_single_query(self, eval_query: EvaluationQuery, query_idx: int, total: int) -> EvaluationResult:
        """Evaluate a single query end-to-end with detailed logging"""
        overall_start = datetime.now()
        
        retrieval_metrics = None
        answer_metrics = None
        generated_answer = ""
        retrieval_latency_ms = 0
        generation_latency_ms = 0
        error = None
        failure_reasons = []
        
        try:
            # ============================================
            # STEP 1: RETRIEVAL EVALUATION
            # ============================================
            retrieval_start = datetime.now()
            
            try:
                retrieval_metrics = await self.retrieval_evaluator.evaluate_retrieval(
                    eval_query.query,
                    self.user_id,
                    k=5
                )
                retrieval_latency_ms = (datetime.now() - retrieval_start).total_seconds() * 1000
                
                if retrieval_metrics.error:
                    failure_reasons.append(f"Retrieval error: {retrieval_metrics.error}")
                    
            except Exception as e:
                retrieval_latency_ms = (datetime.now() - retrieval_start).total_seconds() * 1000
                error_msg = f"Retrieval failed: {str(e)}"
                failure_reasons.append(error_msg)
            
            # ============================================
            # STEP 2: ANSWER GENERATION
            # ============================================
            generation_start = datetime.now()
            
            try:
                initial_state: GraphState = {
                    "query": eval_query.query,
                    "user_id": self.user_id,
                    "answer": None,
                    "verdict": "",
                    "feedback": "",
                    "retries": 0,
                    "trace": [],
                }
                
                final_state = await asyncio.wait_for(
                    app_graph.ainvoke(initial_state),
                    timeout=120.0
                )
                
                generation_latency_ms = (datetime.now() - generation_start).total_seconds() * 1000
                
                if final_state.get("answer"):
                    generated_answer = final_state["answer"].answer
                    citations = [s.citation for s in final_state["answer"].support]

                else:
                    generated_answer = ""
                    citations = []
                    failure_reasons.append("No answer generated")
                
            except asyncio.TimeoutError:
                generation_latency_ms = (datetime.now() - generation_start).total_seconds() * 1000
                error_msg = "Answer generation timed out (120s)"
                failure_reasons.append(error_msg)
                generated_answer = ""
                citations = []
                
            except Exception as e:
                generation_latency_ms = (datetime.now() - generation_start).total_seconds() * 1000
                error_msg = f"Generation failed: {str(e)}"
                failure_reasons.append(error_msg)
                generated_answer = ""
                citations = []
            
            # ============================================
            # STEP 3: ANSWER QUALITY EVALUATION
            # ============================================
            if generated_answer:
                try:

                    results = await retrieve(eval_query.query, self.user_id)
                    retrieved_passages = [p.text for p in results.passages[:5]]
                    
                    answer_metrics = await self.answer_evaluator.evaluate_answer(
                        eval_query.query,
                        generated_answer,
                        eval_query.ground_truth_answer,
                        retrieved_passages,
                        citations
                    )
                    
                    if answer_metrics.error:
                        failure_reasons.append(f"Answer evaluation error: {answer_metrics.error}")
                        
                except Exception as e:
                    error_msg = f"Answer evaluation failed: {str(e)}"
                    failure_reasons.append(error_msg)
            
            # ============================================
            # STEP 4: DETERMINE PASS/FAIL
            # ============================================
            passed = True

            if answer_metrics:
                if answer_metrics.faithfulness_score < 0.7:
                    passed = False
                    failure_reasons.append(f"Low faithfulness: {answer_metrics.faithfulness_score:.2f}")
                
                if answer_metrics.hallucination_detected:
                    passed = False
                    failure_reasons.append("Hallucination detected")
                
                if answer_metrics.relevance_score < 0.7:
                    passed = False
                    failure_reasons.append(f"Low relevance: {answer_metrics.relevance_score:.2f}")
            else:
                passed = False
                failure_reasons.append("Answer metrics not available")
            
            if not generated_answer:
                passed = False
                if "No answer generated" not in failure_reasons:
                    failure_reasons.append("No answer generated")
            
            total_latency_ms = (datetime.now() - overall_start).total_seconds() * 1000

            
            return EvaluationResult(
                query=eval_query.query,
                category=eval_query.category,
                difficulty=eval_query.difficulty,
                retrieval_metrics=retrieval_metrics,
                answer_metrics=answer_metrics,
                generated_answer=generated_answer,
                ground_truth_answer=eval_query.ground_truth_answer,
                retrieval_latency_ms=retrieval_latency_ms,
                generation_latency_ms=generation_latency_ms,
                total_latency_ms=total_latency_ms,
                timestamp=datetime.now().isoformat(),
                passed=passed,
                failure_reasons=failure_reasons,
                error=error
            )
            
        except Exception as e:
            total_latency_ms = (datetime.now() - overall_start).total_seconds() * 1000
            error_msg = f"Overall evaluation failed: {str(e)}"
            traceback.print_exc()
            
            return EvaluationResult(
                query=eval_query.query,
                category=eval_query.category,
                difficulty=eval_query.difficulty,
                retrieval_metrics=retrieval_metrics,
                answer_metrics=answer_metrics,
                generated_answer=generated_answer,
                ground_truth_answer=eval_query.ground_truth_answer,
                retrieval_latency_ms=retrieval_latency_ms,
                generation_latency_ms=generation_latency_ms,
                total_latency_ms=total_latency_ms,
                timestamp=datetime.now().isoformat(),
                passed=False,
                failure_reasons=[error_msg] + failure_reasons,
                error=error_msg
            )
    
    async def evaluate_dataset(
        self, 
        dataset: EvaluationDataset,
        max_concurrent: int = 1
    ) -> List[EvaluationResult]:
        """Evaluate entire dataset with sequential processing for reliability"""
        total = len(dataset.queries)
        self.results = []

        
        if max_concurrent == 1:

            for idx, query in enumerate(dataset.queries, 1):
                try:

                    result = await self.evaluate_single_query(query, idx, total)
                    self.results.append(result)

                except Exception as e:

                    traceback.print_exc()
                    self.results.append(EvaluationResult(
                        query=query.query,
                        category=query.category,
                        difficulty=query.difficulty,
                        retrieval_metrics=None,
                        answer_metrics=None,
                        generated_answer="",
                        ground_truth_answer=query.ground_truth_answer,
                        retrieval_latency_ms=0,
                        generation_latency_ms=0,
                        total_latency_ms=0,
                        timestamp=datetime.now().isoformat(),
                        passed=False,
                        failure_reasons=[f"Complete failure: {str(e)}"],
                        error=str(e)
                    ))
        else:

            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def evaluate_with_semaphore(query, idx):
                async with semaphore:
                    try:
                        return await self.evaluate_single_query(query, idx, total)
                    except Exception as e:

                        traceback.print_exc()
                        return EvaluationResult(
                            query=query.query,
                            category=query.category,
                            difficulty=query.difficulty,
                            retrieval_metrics=None,
                            answer_metrics=None,
                            generated_answer="",
                            ground_truth_answer=query.ground_truth_answer,
                            retrieval_latency_ms=0,
                            generation_latency_ms=0,
                            total_latency_ms=0,
                            timestamp=datetime.now().isoformat(),
                            passed=False,
                            failure_reasons=[f"Semaphore failure: {str(e)}"],
                            error=str(e)
                        )
            
            tasks = [evaluate_with_semaphore(q, i+1) for i, q in enumerate(dataset.queries)]
            self.results = await asyncio.gather(*tasks, return_exceptions=False)
        
        return self.results
    
    def generate_report(self, output_path: str = "evaluation_report.json"):
        """Generate comprehensive evaluation report"""
        if not self.results:
            return None

        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        pass_rate = passed / total if total > 0 else 0
        
        valid_results = [r for r in self.results if r.answer_metrics]
        
        if valid_results:
            avg_faithfulness = sum(r.answer_metrics.faithfulness_score for r in valid_results) / len(valid_results)
            avg_relevance = sum(r.answer_metrics.relevance_score for r in valid_results) / len(valid_results)
            avg_citation_accuracy = sum(r.answer_metrics.citation_accuracy for r in valid_results) / len(valid_results)
            avg_completeness = sum(r.answer_metrics.completeness_score for r in valid_results) / len(valid_results)
            hallucination_rate = sum(1 for r in valid_results if r.answer_metrics.hallucination_detected) / len(valid_results)
        else:
            avg_faithfulness = avg_relevance = avg_citation_accuracy = avg_completeness = 0.0
            hallucination_rate = 1.0
        
        avg_retrieval_latency = sum(r.retrieval_latency_ms for r in self.results) / total
        avg_generation_latency = sum(r.generation_latency_ms for r in self.results) / total
        avg_total_latency = sum(r.total_latency_ms for r in self.results) / total
        
        categories = {}
        for result in self.results:
            if result.category not in categories:
                categories[result.category] = {"total": 0, "passed": 0}
            categories[result.category]["total"] += 1
            if result.passed:
                categories[result.category]["passed"] += 1
        
        for cat in categories:
            categories[cat]["pass_rate"] = categories[cat]["passed"] / categories[cat]["total"]
        
        report = {
            "summary": {
                "total_queries": total,
                "valid_evaluations": len(valid_results),
                "passed": passed,
                "failed": total - passed,
                "pass_rate": pass_rate,
                "avg_retrieval_latency_ms": avg_retrieval_latency,
                "avg_generation_latency_ms": avg_generation_latency,
                "avg_total_latency_ms": avg_total_latency
            },
            "answer_metrics": {
                "avg_faithfulness": avg_faithfulness,
                "avg_relevance": avg_relevance,
                "avg_citation_accuracy": avg_citation_accuracy,
                "avg_completeness": avg_completeness,
                "hallucination_rate": hallucination_rate
            },
            "category_breakdown": categories,
            "detailed_results": [asdict(r) for r in self.results]
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
