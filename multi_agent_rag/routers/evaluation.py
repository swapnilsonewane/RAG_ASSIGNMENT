import asyncio
import traceback
from typing import Optional, List
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from multi_agent_rag.core.security import get_current_user, require_roles
from multi_agent_rag.core.database import get_db
from multi_agent_rag.core.audit import log_audit
from multi_agent_rag.services.evaluation_service import (
    RAGEvaluationPipeline,
    EvaluationDataset,
    create_sample_dataset,
    EvaluationQuery
)

# ==================================================
# ROUTER
# ==================================================

router = APIRouter(prefix="/evaluation", tags=["evaluation"])

# ==================================================
# GLOBAL STATE FOR EVALUATION
# ==================================================

evaluation_runs = {}


# ==================================================
# SCHEMAS
# ==================================================

class EvaluationRunRequest(BaseModel):
    dataset_name: str = Field(..., description="Name of the evaluation dataset")
    max_concurrent: int = Field(default=1, ge=1, le=5)
    user_id_override: Optional[str] = None


class EvaluationRunResponse(BaseModel):
    run_id: str
    status: str
    message: str


class EvaluationStatusResponse(BaseModel):
    run_id: str
    status: str
    progress: Optional[dict] = None
    results: Optional[dict] = None
    error: Optional[str] = None
    started_at: str
    completed_at: Optional[str] = None


class AddQueryRequest(BaseModel):
    query: str
    ground_truth_answer: str
    category: str
    difficulty: str
    expected_citations: List[str]


class DatasetResponse(BaseModel):
    name: str
    total_queries: int
    categories: dict
    queries: List[dict]


# ==================================================
# BACKGROUND TASK FOR EVALUATION
# ==================================================

async def run_evaluation_task(
    run_id: str,
    dataset: EvaluationDataset,
    user_id: str,
    max_concurrent: int
):
    """Background task to run evaluation with comprehensive error handling"""
    try:

        evaluation_runs[run_id]["status"] = "running"
        evaluation_runs[run_id]["current_query"] = 0
        
        Path("reports").mkdir(exist_ok=True)

        pipeline = RAGEvaluationPipeline(user_id=user_id)

        results = await pipeline.evaluate_dataset(dataset, max_concurrent=max_concurrent)
        
        report_path = f"reports/evaluation_{run_id}.json"

        report = pipeline.generate_report(report_path)
        
        if report:
            evaluation_runs[run_id]["status"] = "completed"
            evaluation_runs[run_id]["completed_at"] = datetime.now().isoformat()
            evaluation_runs[run_id]["results"] = report
            evaluation_runs[run_id]["total_processed"] = len(results)

        else:
            evaluation_runs[run_id]["status"] = "failed"
            evaluation_runs[run_id]["error"] = "Failed to generate report"
            evaluation_runs[run_id]["completed_at"] = datetime.now().isoformat()

        
    except asyncio.CancelledError:
        error_msg = "Evaluation task was cancelled"
        evaluation_runs[run_id]["status"] = "cancelled"
        evaluation_runs[run_id]["error"] = error_msg
        evaluation_runs[run_id]["completed_at"] = datetime.now().isoformat()
        
    except Exception as e:
        error_msg = f"Evaluation failed: {str(e)}"
        traceback.print_exc()
        evaluation_runs[run_id]["status"] = "failed"
        evaluation_runs[run_id]["error"] = error_msg
        evaluation_runs[run_id]["completed_at"] = datetime.now().isoformat()
        evaluation_runs[run_id]["traceback"] = traceback.format_exc()


# ==================================================
# ENDPOINTS
# ==================================================

@router.post(
    "/run",
    response_model=EvaluationRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_roles(["admin"]))],
)
async def start_evaluation_run(
    request: EvaluationRunRequest,
    background_tasks: BackgroundTasks,
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Start an evaluation run in the background
    
    Only admin users can run evaluations
    """
    try:

        run_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if request.dataset_name == "sample":
            dataset = create_sample_dataset()

        else:

            try:
                dataset = EvaluationDataset()
                dataset_path = f"datasets/{request.dataset_name}.json"
                dataset.load_from_json(dataset_path)

            except FileNotFoundError:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Dataset '{request.dataset_name}' not found"
                )
        

        evaluation_runs[run_id] = {
            "status": "pending",
            "started_at": datetime.now().isoformat(),
            "completed_at": None,
            "dataset_name": request.dataset_name,
            "total_queries": len(dataset.queries),
            "current_query": 0,
            "total_processed": 0,
            "user_id": user.user_id,
            "error": None
        }
        
        user_id_for_eval = user.user_id
        
        asyncio.create_task(
            run_evaluation_task(
                run_id,
                dataset,
                user_id_for_eval,
                request.max_concurrent
            )
        )

        await log_audit(
            db,
            user.user_id,
            action="START_EVALUATION",
            resource="evaluation",
            metadata={
                "run_id": run_id,
                "dataset_name": request.dataset_name,
                "total_queries": len(dataset.queries),
                "max_concurrent": request.max_concurrent
            },
        )
        
        return EvaluationRunResponse(
            run_id=run_id,
            status="pending",
            message=f"Evaluation run started with {len(dataset.queries)} queries (max_concurrent={request.max_concurrent})"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start evaluation: {str(e)}"
        )


@router.get(
    "/run/{run_id}",
    response_model=EvaluationStatusResponse,
    dependencies=[Depends(require_roles(["admin"]))],
)
async def get_evaluation_status(
    run_id: str,
    user=Depends(get_current_user),
):
    """
    Get the status and complete results of an evaluation run
    """
    if run_id not in evaluation_runs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evaluation run '{run_id}' not found"
        )
    
    run_data = evaluation_runs[run_id]
    
    total = run_data.get("total_queries", 0)
    processed = run_data.get("total_processed", run_data.get("current_query", 0))
    
    progress = {
        "total_queries": total,
        "processed_queries": processed,
        "progress_percentage": (processed / total * 100) if total > 0 else 0,
        "dataset_name": run_data.get("dataset_name"),
        "current_status": run_data.get("status")
    }
    
    return EvaluationStatusResponse(
        run_id=run_id,
        status=run_data["status"],
        progress=progress,
        results=run_data.get("results"),
        error=run_data.get("error"),
        started_at=run_data["started_at"],
        completed_at=run_data.get("completed_at")
    )


@router.get(
    "/runs",
    dependencies=[Depends(require_roles(["admin"]))],
)
async def list_evaluation_runs(
    user=Depends(get_current_user),
):
    """
    List all evaluation runs
    """
    runs = [
        {
            "run_id": run_id,
            "status": data["status"],
            "started_at": data["started_at"],
            "completed_at": data.get("completed_at"),
            "dataset_name": data.get("dataset_name"),
            "total_queries": data.get("total_queries"),
            "processed_queries": data.get("total_processed", 0),
            "error": data.get("error")
        }
        for run_id, data in sorted(
            evaluation_runs.items(),
            key=lambda x: x[1]["started_at"],
            reverse=True
        )
    ]
    
    return {
        "runs": runs,
        "total_runs": len(runs),
        "active_runs": len([r for r in runs if r["status"] in ["pending", "running"]]),
        "completed_runs": len([r for r in runs if r["status"] == "completed"]),
        "failed_runs": len([r for r in runs if r["status"] == "failed"])
    }


@router.get(
    "/datasets/{dataset_name}",
    response_model=DatasetResponse,
    dependencies=[Depends(require_roles(["admin"]))],
)
async def get_dataset(
    dataset_name: str,
    user=Depends(get_current_user),
):
    """
    Retrieve an evaluation dataset
    """
    try:
        if dataset_name == "sample":
            dataset = create_sample_dataset()
        else:
            dataset = EvaluationDataset()
            dataset.load_from_json(f"datasets/{dataset_name}.json")
        
        categories = {}
        for query in dataset.queries:
            categories[query.category] = categories.get(query.category, 0) + 1
        
        return DatasetResponse(
            name=dataset_name,
            total_queries=len(dataset.queries),
            categories=categories,
            queries=[
                {
                    "query": q.query,
                    "category": q.category,
                    "difficulty": q.difficulty,
                    "expected_citations": q.expected_citations
                }
                for q in dataset.queries
            ]
        )
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset '{dataset_name}' not found"
        )


@router.post(
    "/datasets/{dataset_name}/queries",
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_roles(["admin"]))],
)
async def add_query_to_dataset(
    dataset_name: str,
    request: AddQueryRequest,
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Add a new query to an evaluation dataset
    """
    try:

        Path("datasets").mkdir(exist_ok=True)
        
        dataset = EvaluationDataset()
        dataset_path = f"datasets/{dataset_name}.json"
        
        try:
            dataset.load_from_json(dataset_path)
        except FileNotFoundError:
            pass  

        eval_query = EvaluationQuery(
            query=request.query,
            ground_truth_answer=request.ground_truth_answer,
            category=request.category,
            difficulty=request.difficulty,
            expected_citations=request.expected_citations
        )
        dataset.add_query(eval_query)
        
        dataset.save_to_json(dataset_path)

        await log_audit(
            db,
            user.user_id,
            action="ADD_EVAL_QUERY",
            resource="evaluation",
            metadata={
                "dataset_name": dataset_name,
                "query": request.query,
                "category": request.category
            },
        )
        
        return {
            "message": "Query added successfully",
            "dataset_name": dataset_name,
            "total_queries": len(dataset.queries)
        }
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add query: {str(e)}"
        )


@router.get(
    "/metrics/summary",
    dependencies=[Depends(require_roles(["admin"]))],
)
async def get_metrics_summary(
    user=Depends(get_current_user),
):
    """
    Get a summary of evaluation metrics across all completed runs
    """
    completed_runs = [
        (run_id, data)
        for run_id, data in evaluation_runs.items()
        if data["status"] == "completed" and data.get("results")
    ]
    
    if not completed_runs:
        return {
            "message": "No completed evaluation runs found",
            "total_runs": 0
        }

    total_runs = len(completed_runs)
    
    avg_pass_rate = sum(
        data["results"]["summary"]["pass_rate"]
        for _, data in completed_runs
    ) / total_runs
    
    avg_faithfulness = sum(
        data["results"]["answer_metrics"]["avg_faithfulness"]
        for _, data in completed_runs
    ) / total_runs
    
    avg_relevance = sum(
        data["results"]["answer_metrics"]["avg_relevance"]
        for _, data in completed_runs
    ) / total_runs
    
    avg_retrieval_latency = sum(
        data["results"]["summary"]["avg_retrieval_latency_ms"]
        for _, data in completed_runs
    ) / total_runs
    
    avg_generation_latency = sum(
        data["results"]["summary"]["avg_generation_latency_ms"]
        for _, data in completed_runs
    ) / total_runs
    
    return {
        "total_runs": total_runs,
        "avg_pass_rate": avg_pass_rate,
        "avg_faithfulness": avg_faithfulness,
        "avg_relevance": avg_relevance,
        "avg_retrieval_latency_ms": avg_retrieval_latency,
        "avg_generation_latency_ms": avg_generation_latency,
        "recent_runs": [
            {
                "run_id": run_id,
                "completed_at": data["completed_at"],
                "pass_rate": data["results"]["summary"]["pass_rate"],
                "total_queries": data["total_queries"],
                "valid_evaluations": data["results"]["summary"]["valid_evaluations"],
                "avg_retrieval_latency_ms": data["results"]["summary"]["avg_retrieval_latency_ms"],
                "avg_generation_latency_ms": data["results"]["summary"]["avg_generation_latency_ms"],
                "avg_faithfulness": data["results"]["answer_metrics"]["avg_faithfulness"],
                "avg_relevance": data["results"]["answer_metrics"]["avg_relevance"],
                "hallucination_rate": data["results"]["answer_metrics"]["hallucination_rate"]
            }
            for run_id, data in completed_runs[-5:]
        ]
    }


@router.delete(
    "/run/{run_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(require_roles(["admin"]))],
)
async def delete_evaluation_run(
    run_id: str,
    user=Depends(get_current_user),
):
    """
    Delete an evaluation run from memory
    """
    if run_id not in evaluation_runs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evaluation run '{run_id}' not found"
        )
    
    del evaluation_runs[run_id]
    return None


@router.get(
    "/run/{run_id}/errors",
    dependencies=[Depends(require_roles(["admin"]))],
)
async def get_evaluation_errors(
    run_id: str,
    user=Depends(get_current_user),
):
    """
    Get detailed error information for a failed evaluation run
    """
    if run_id not in evaluation_runs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evaluation run '{run_id}' not found"
        )
    
    run_data = evaluation_runs[run_id]

    failed_queries = []
    if run_data.get("results") and run_data["results"].get("detailed_results"):
        failed_queries = [
            {
                "query": r["query"],
                "category": r["category"],
                "failure_reasons": r["failure_reasons"],
                "error": r.get("error"),
                "generated_answer": r.get("generated_answer", "")[:200],
                "answer_metrics": r.get("answer_metrics")
            }
            for r in run_data["results"]["detailed_results"]
            if not r["passed"]
        ]
    
    return {
        "run_id": run_id,
        "status": run_data["status"],
        "error": run_data.get("error"),
        "traceback": run_data.get("traceback"),
        "failed_queries": failed_queries,
        "total_failures": len(failed_queries)
    }


@router.get(
    "/run/{run_id}/report",
    dependencies=[Depends(require_roles(["admin"]))],
)
async def get_full_evaluation_report(
    run_id: str,
    user=Depends(get_current_user),
):
    """
    Get the complete evaluation report with all details
    
    Returns:
    - Summary metrics (pass rate, latencies)
    - Answer quality metrics (faithfulness, relevance, etc.)
    - Category breakdown
    - Detailed results for each query
    """
    if run_id not in evaluation_runs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evaluation run '{run_id}' not found"
        )
    
    run_data = evaluation_runs[run_id]
    
    if run_data["status"] != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Evaluation run is not completed. Current status: {run_data['status']}"
        )
    
    if not run_data.get("results"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found for this evaluation run"
        )
    

    return {
        "run_id": run_id,
        "dataset_name": run_data["dataset_name"],
        "started_at": run_data["started_at"],
        "completed_at": run_data["completed_at"],
        "report": run_data["results"]
    }