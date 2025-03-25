from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, Type, TypeVar, Union
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import csv
import os
from pathlib import Path

import logfire
import pandas as pd
from pydantic import BaseModel, Field

from llm_classifiers.classifier import Classifier, Example


T = TypeVar('T', bound=BaseModel)


class EvaluationFormat(str, Enum):
    """Format options for evaluation results."""
    JSON = "json"
    CSV = "csv"
    DATAFRAME = "dataframe"
    SUMMARY = "summary"


@dataclass
class EvaluationResult(Generic[T]):
    """Result of a single evaluation."""
    input_text: str
    ground_truth: T
    prediction: T
    correct: bool
    metadata: Dict[str, Any] = field(default_factory=dict)
    latency_ms: Optional[float] = None


@dataclass
class EvaluationMetrics:
    """Metrics computed from evaluation results."""
    total_examples: int
    correct_predictions: int
    accuracy: float
    average_latency_ms: Optional[float] = None
    class_metrics: Optional[Dict[str, Dict[str, float]]] = None
    field_level_accuracy: Optional[Dict[str, float]] = None
    confusion_matrix: Optional[Dict[str, Dict[str, int]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        """Return a string representation of the metrics."""
        result = [
            f"Total examples: {self.total_examples}",
            f"Correct predictions: {self.correct_predictions}",
            f"Accuracy: {self.accuracy:.4f}",
        ]
        
        if self.average_latency_ms is not None:
            result.append(f"Average latency: {self.average_latency_ms:.2f}ms")
        
        if self.field_level_accuracy:
            result.append("\nField-level accuracy:")
            for field, acc in self.field_level_accuracy.items():
                result.append(f"  {field}: {acc:.4f}")
        
        if self.class_metrics:
            result.append("\nClass metrics:")
            for cls, metrics in self.class_metrics.items():
                result.append(f"  {cls}:")
                for metric_name, value in metrics.items():
                    result.append(f"    {metric_name}: {value:.4f}")
        
        return "\n".join(result)


class ClassifierEvaluator(Generic[T]):
    """
    Evaluator for classifier performance.
    
    This class provides tools to evaluate classifiers against datasets
    and compare results across different configurations.
    """
    
    def __init__(
        self,
        classifier: Classifier,
        comparison_field: Optional[str] = None,
        exact_match: bool = False,
        max_workers: int = 4,
        enable_logging: bool = True
    ):
        """
        Initialize an evaluator.
        
        Args:
            classifier: The classifier to evaluate.
            comparison_field: The field to use for correctness comparison.
                If None, all fields are compared.
            exact_match: Whether to require exact match for all fields.
            max_workers: Maximum number of parallel workers for evaluation.
            enable_logging: Whether to enable logfire logging.
        """
        self.classifier = classifier
        self.comparison_field = comparison_field
        self.exact_match = exact_match
        self.max_workers = max_workers
        self.enable_logging = enable_logging
        
        if self.enable_logging:
            # Initialize logfire if not already initialized
            try:
                logfire.init(service_name="llm-classifier-eval")
            except Exception:
                # Logfire might be already initialized, continue silently
                pass
            
            # Log initialization
            logfire.log(
                "ClassifierEvaluator initialized",
                model_name=getattr(classifier.model, "model_name", str(classifier.model)),
                comparison_field=comparison_field,
                exact_match=exact_match
            )
    
    @logfire.trace
    def _check_correctness(self, ground_truth: T, prediction: T) -> bool:
        """
        Check if a prediction is correct based on configured comparison options.
        
        Args:
            ground_truth: The correct answer.
            prediction: The classifier's prediction.
            
        Returns:
            Whether the prediction is correct according to comparison rules.
        """
        if self.exact_match:
            # For exact match, compare all fields
            truth_dict = ground_truth.model_dump()
            pred_dict = prediction.model_dump()
            return truth_dict == pred_dict
        
        elif self.comparison_field:
            # Compare only the specified field
            truth_val = getattr(ground_truth, self.comparison_field)
            pred_val = getattr(prediction, self.comparison_field)
            return truth_val == pred_val
        
        else:
            # Default: compare all fields individually
            truth_dict = ground_truth.model_dump()
            pred_dict = prediction.model_dump()
            
            # Check for any match
            for field, value in truth_dict.items():
                if field in pred_dict and pred_dict[field] == value:
                    return True
            return False
    
    @logfire.trace
    def _evaluate_single(
        self, 
        example: Example,
        include_latency: bool = True
    ) -> EvaluationResult:
        """
        Evaluate the classifier on a single example.
        
        Args:
            example: The example to evaluate.
            include_latency: Whether to measure and include latency.
            
        Returns:
            Evaluation result for this example.
        """
        start_time = time.time() if include_latency else None
        
        # Run classification
        prediction = self.classifier.classify(example.input_text)
        
        # Calculate latency if requested
        latency_ms = None
        if include_latency and start_time is not None:
            latency_ms = (time.time() - start_time) * 1000
        
        # Check correctness
        correct = self._check_correctness(example.output, prediction)
        
        # Create and return result
        return EvaluationResult(
            input_text=example.input_text,
            ground_truth=example.output,
            prediction=prediction,
            correct=correct,
            latency_ms=latency_ms
        )
    
    @logfire.trace
    def evaluate(
        self,
        test_data: List[Example],
        include_latency: bool = True,
        parallel: bool = True
    ) -> List[EvaluationResult]:
        """
        Evaluate the classifier on a list of examples.
        
        Args:
            test_data: List of examples to evaluate.
            include_latency: Whether to measure and include latency.
            parallel: Whether to run evaluations in parallel.
            
        Returns:
            List of evaluation results.
        """
        results = []
        
        if self.enable_logging:
            logfire.log(
                "Starting evaluation",
                test_size=len(test_data),
                parallel=parallel,
                include_latency=include_latency
            )
        
        if parallel and len(test_data) > 1:
            # Parallel evaluation using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_example = {
                    executor.submit(self._evaluate_single, example, include_latency): example
                    for example in test_data
                }
                
                for future in as_completed(future_to_example):
                    results.append(future.result())
        else:
            # Sequential evaluation
            for example in test_data:
                results.append(self._evaluate_single(example, include_latency))
        
        if self.enable_logging:
            logfire.log("Evaluation completed", results_count=len(results))
        
        return results
    
    @logfire.trace
    def compute_metrics(
        self, 
        results: List[EvaluationResult]
    ) -> EvaluationMetrics:
        """
        Compute metrics from evaluation results.
        
        Args:
            results: List of evaluation results.
            
        Returns:
            Computed metrics.
        """
        if not results:
            return EvaluationMetrics(
                total_examples=0,
                correct_predictions=0,
                accuracy=0.0
            )
        
        # Basic metrics
        total = len(results)
        correct = sum(1 for r in results if r.correct)
        accuracy = correct / total if total > 0 else 0
        
        # Calculate average latency if available
        latencies = [r.latency_ms for r in results if r.latency_ms is not None]
        avg_latency = sum(latencies) / len(latencies) if latencies else None
        
        # Compute field-level accuracy
        field_accuracy = {}
        if results and hasattr(results[0].ground_truth, "model_dump"):
            # Get all fields from the first result
            fields = results[0].ground_truth.model_dump().keys()
            
            for field in fields:
                field_correct = 0
                for result in results:
                    if hasattr(result.ground_truth, field) and hasattr(result.prediction, field):
                        truth_val = getattr(result.ground_truth, field)
                        pred_val = getattr(result.prediction, field)
                        if truth_val == pred_val:
                            field_correct += 1
                
                field_accuracy[field] = field_correct / total if total > 0 else 0
        
        # Compute class-specific metrics if comparison field is set
        class_metrics = None
        confusion_matrix = None
        
        if self.comparison_field and results:
            # Get unique classes from ground truth
            classes = set()
            for result in results:
                if hasattr(result.ground_truth, self.comparison_field):
                    classes.add(getattr(result.ground_truth, self.comparison_field))
            
            # Initialize confusion matrix
            confusion_matrix = {
                str(true_class): {str(pred_class): 0 for pred_class in classes}
                for true_class in classes
            }
            
            # Populate confusion matrix
            for result in results:
                true_class = getattr(result.ground_truth, self.comparison_field)
                pred_class = getattr(result.prediction, self.comparison_field)
                confusion_matrix[str(true_class)][str(pred_class)] += 1
            
            # Compute per-class metrics
            class_metrics = {}
            for cls in classes:
                cls_str = str(cls)
                
                # Get TP, FP, TN, FN for this class
                tp = confusion_matrix[cls_str][cls_str]
                fp = sum(confusion_matrix[other][cls_str] for other in confusion_matrix if other != cls_str)
                fn = sum(confusion_matrix[cls_str][other] for other in confusion_matrix[cls_str] if other != cls_str)
                
                # Compute precision, recall, f1
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                class_metrics[cls_str] = {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "support": tp + fn
                }
        
        return EvaluationMetrics(
            total_examples=total,
            correct_predictions=correct,
            accuracy=accuracy,
            average_latency_ms=avg_latency,
            class_metrics=class_metrics,
            field_level_accuracy=field_accuracy,
            confusion_matrix=confusion_matrix
        )
    
    @logfire.trace
    def export_results(
        self,
        results: List[EvaluationResult],
        format: EvaluationFormat = EvaluationFormat.SUMMARY,
        file_path: Optional[str] = None,
        include_metrics: bool = True
    ) -> Union[str, pd.DataFrame, None]:
        """
        Export evaluation results in various formats.
        
        Args:
            results: List of evaluation results.
            format: The format to export results in.
            file_path: Optional path to save results to.
            include_metrics: Whether to include computed metrics.
            
        Returns:
            Exported results as string or DataFrame, or None if saved to file.
        """
        # Compute metrics if requested
        metrics = self.compute_metrics(results) if include_metrics else None
        
        # Convert results to serializable format
        serialized_results = []
        for result in results:
            item = {
                "input_text": result.input_text,
                "ground_truth": result.ground_truth.model_dump() if hasattr(result.ground_truth, "model_dump") else str(result.ground_truth),
                "prediction": result.prediction.model_dump() if hasattr(result.prediction, "model_dump") else str(result.prediction),
                "correct": result.correct
            }
            
            if result.latency_ms is not None:
                item["latency_ms"] = result.latency_ms
            
            if result.metadata:
                item["metadata"] = result.metadata
                
            serialized_results.append(item)
        
        # Prepare metrics if available
        metrics_dict = None
        if metrics:
            metrics_dict = {
                "total_examples": metrics.total_examples,
                "correct_predictions": metrics.correct_predictions,
                "accuracy": metrics.accuracy
            }
            
            if metrics.average_latency_ms is not None:
                metrics_dict["average_latency_ms"] = metrics.average_latency_ms
            
            if metrics.field_level_accuracy:
                metrics_dict["field_level_accuracy"] = metrics.field_level_accuracy
            
            if metrics.class_metrics:
                metrics_dict["class_metrics"] = metrics.class_metrics
            
            if metrics.confusion_matrix:
                metrics_dict["confusion_matrix"] = metrics.confusion_matrix
                
            if metrics.metadata:
                metrics_dict["metadata"] = metrics.metadata
        
        # Format the results
        if format == EvaluationFormat.JSON:
            output = {
                "results": serialized_results
            }
            if metrics_dict:
                output["metrics"] = metrics_dict
                
            output_str = json.dumps(output, indent=2)
            
            if file_path:
                os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
                with open(file_path, "w") as f:
                    f.write(output_str)
                return None
            
            return output_str
            
        elif format == EvaluationFormat.CSV:
            # Flatten results for CSV
            flat_results = []
            for result in serialized_results:
                flat_result = {
                    "input_text": result["input_text"],
                    "correct": result["correct"]
                }
                
                # Flatten ground truth and prediction
                if isinstance(result["ground_truth"], dict):
                    for key, value in result["ground_truth"].items():
                        flat_result[f"ground_truth_{key}"] = value
                else:
                    flat_result["ground_truth"] = result["ground_truth"]
                
                if isinstance(result["prediction"], dict):
                    for key, value in result["prediction"].items():
                        flat_result[f"prediction_{key}"] = value
                else:
                    flat_result["prediction"] = result["prediction"]
                
                # Add latency if available
                if "latency_ms" in result:
                    flat_result["latency_ms"] = result["latency_ms"]
                
                flat_results.append(flat_result)
            
            if file_path:
                os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
                with open(file_path, "w", newline="") as f:
                    if flat_results:
                        writer = csv.DictWriter(f, fieldnames=flat_results[0].keys())
                        writer.writeheader()
                        writer.writerows(flat_results)
                return None
            
            # Return DataFrame instead of CSV string if no file path
            return pd.DataFrame(flat_results)
        
        elif format == EvaluationFormat.DATAFRAME:
            # Create DataFrame
            df = pd.json_normalize(serialized_results)
            
            # Save to file if path provided
            if file_path:
                os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
                
                # Determine format based on file extension
                ext = os.path.splitext(file_path)[1].lower()
                if ext == ".csv":
                    df.to_csv(file_path, index=False)
                elif ext == ".xlsx":
                    df.to_excel(file_path, index=False)
                elif ext == ".parquet":
                    df.to_parquet(file_path, index=False)
                else:  # Default to CSV
                    df.to_csv(file_path, index=False)
                
                # Save metrics to a separate file if available
                if metrics_dict and file_path:
                    base, ext = os.path.splitext(file_path)
                    metrics_path = f"{base}_metrics{ext}"
                    metrics_df = pd.json_normalize(metrics_dict)
                    
                    if ext == ".csv":
                        metrics_df.to_csv(metrics_path, index=False)
                    elif ext == ".xlsx":
                        metrics_df.to_excel(metrics_path, index=False)
                    elif ext == ".parquet":
                        metrics_df.to_parquet(metrics_path, index=False)
                    else:  # Default to CSV
                        metrics_df.to_csv(metrics_path, index=False)
                
                return None
            
            return df
        
        elif format == EvaluationFormat.SUMMARY:
            # Generate text summary
            if metrics:
                summary = str(metrics)
                
                if file_path:
                    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
                    with open(file_path, "w") as f:
                        f.write(summary)
                    return None
                
                return summary
            
            return "No metrics available for summary."
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    @logfire.trace
    def compare_classifiers(
        self,
        classifiers: List[Classifier],
        test_data: List[Example],
        names: Optional[List[str]] = None,
        include_latency: bool = True
    ) -> pd.DataFrame:
        """
        Compare multiple classifiers on the same test data.
        
        Args:
            classifiers: List of classifiers to compare.
            test_data: List of examples to evaluate.
            names: Optional list of names for the classifiers.
            include_latency: Whether to measure and include latency.
            
        Returns:
            DataFrame with comparison results.
        """
        if names and len(names) != len(classifiers):
            raise ValueError("Number of names must match number of classifiers")
        
        names = names or [f"Classifier {i+1}" for i in range(len(classifiers))]
        
        comparison_results = []
        
        for i, classifier in enumerate(classifiers):
            name = names[i]
            
            # Create evaluator for this classifier
            evaluator = ClassifierEvaluator(
                classifier=classifier,
                comparison_field=self.comparison_field,
                exact_match=self.exact_match,
                max_workers=self.max_workers,
                enable_logging=self.enable_logging
            )
            
            # Evaluate
            results = evaluator.evaluate(test_data, include_latency=include_latency)
            metrics = evaluator.compute_metrics(results)
            
            # Add to comparison
            comparison_results.append({
                "name": name,
                "accuracy": metrics.accuracy,
                "correct_predictions": metrics.correct_predictions,
                "total_examples": metrics.total_examples,
                "average_latency_ms": metrics.average_latency_ms,
                "field_level_accuracy": metrics.field_level_accuracy,
                "class_metrics": metrics.class_metrics
            })
        
        if self.enable_logging:
            logfire.log(
                "Classifier comparison completed", 
                classifiers_count=len(classifiers),
                test_data_size=len(test_data)
            )
        
        # Return as DataFrame
        return pd.DataFrame(comparison_results)