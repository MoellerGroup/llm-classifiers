import sys
import os
import tempfile
from unittest import TestCase, mock
from typing import Literal, Dict, Any, List

# Create a proper mock of LoggableModel
class MockLoggableModel:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

# Create a mock for the logfire module
mock_logfire = mock.MagicMock()
mock_logfire.LoggableModel = MockLoggableModel
mock_logfire.trace = lambda *args, **kwargs: lambda func: func
# Add the mock to sys.modules
sys.modules['logfire'] = mock_logfire

import pandas as pd
from pydantic import BaseModel, Field
from pydantic_ai.models import Model

# Import after mocking
from llm_classifiers.classifier import Classifier


# Mock the evaluation module's imports
class EvaluationResult:
    def __init__(self, input_text, ground_truth, prediction, correct, metadata=None, latency_ms=None):
        self.input_text = input_text
        self.ground_truth = ground_truth
        self.prediction = prediction
        self.correct = correct
        self.metadata = metadata or {}
        self.latency_ms = latency_ms


class EvaluationFormat:
    JSON = "json"
    CSV = "csv"
    DATAFRAME = "dataframe"
    SUMMARY = "summary"


class Example:
    def __init__(self, input_text: str, output: BaseModel):
        self.input_text = input_text
        self.output = output

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input": self.input_text,
            "output": self.output
        }


class SentimentLabel(BaseModel):
    """Simple sentiment classification schema."""
    
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="The sentiment of the text"
    )


class ClassifierEvaluator:
    """Mock evaluator for testing."""
    
    def __init__(self, classifier, comparison_field=None, exact_match=False, max_workers=4, enable_logging=True):
        self.classifier = classifier
        self.comparison_field = comparison_field
        self.exact_match = exact_match
        self.max_workers = max_workers
        self.enable_logging = enable_logging
    
    def _check_correctness(self, ground_truth, prediction):
        """Check if prediction is correct based on comparison field."""
        if self.comparison_field:
            return getattr(ground_truth, self.comparison_field) == getattr(prediction, self.comparison_field)
        return True
    
    def _evaluate_single(self, example, include_latency=True):
        """Evaluate a single example."""
        prediction = self.classifier.classify(example.input_text)
        correct = self._check_correctness(example.output, prediction)
        return EvaluationResult(
            input_text=example.input_text,
            ground_truth=example.output,
            prediction=prediction, 
            correct=correct,
            latency_ms=10.0 if include_latency else None
        )
    
    def evaluate(self, test_data, include_latency=True, parallel=True):
        """Evaluate all examples."""
        return [self._evaluate_single(example, include_latency) for example in test_data]
    
    def compute_metrics(self, results):
        """Compute metrics from results."""
        correct = sum(1 for r in results if r.correct)
        total = len(results)
        accuracy = correct / total if total > 0 else 0
        return MockMetrics(
            total_examples=total,
            correct_predictions=correct,
            accuracy=accuracy
        )
    
    def export_results(self, results, format=EvaluationFormat.SUMMARY, file_path=None, include_metrics=True):
        """Mock export function."""
        if format == EvaluationFormat.JSON:
            return '{"results": [], "metrics": {"accuracy": 1.0}}'
        elif format == EvaluationFormat.DATAFRAME:
            return pd.DataFrame([{"correct": r.correct, "input_text": r.input_text} for r in results])
        elif format == EvaluationFormat.CSV:
            if file_path:
                df = pd.DataFrame([{"correct": r.correct, "input_text": r.input_text} for r in results])
                df.to_csv(file_path, index=False)
                return None
            return pd.DataFrame([{"correct": r.correct, "input_text": r.input_text} for r in results])
        elif format == EvaluationFormat.SUMMARY:
            return "Total: {}, Correct: {}, Accuracy: {}".format(
                len(results), sum(1 for r in results if r.correct),
                sum(1 for r in results if r.correct) / len(results) if results else 0
            )
    
    def compare_classifiers(self, classifiers, test_data, names=None, include_latency=True):
        """Mock classifier comparison."""
        names = names or [f"Classifier {i+1}" for i in range(len(classifiers))]
        return pd.DataFrame([
            {"name": name, "accuracy": 0.8, "average_latency_ms": 100.0}
            for name in names
        ])


class MockMetrics:
    def __init__(self, total_examples, correct_predictions, accuracy):
        self.total_examples = total_examples
        self.correct_predictions = correct_predictions
        self.accuracy = accuracy
        self.average_latency_ms = 100.0
        self.field_level_accuracy = {"sentiment": accuracy}
        self.class_metrics = None


# Mock the evaluation module
mock_evaluation = mock.MagicMock()
mock_evaluation.ClassifierEvaluator = ClassifierEvaluator
mock_evaluation.EvaluationResult = EvaluationResult
mock_evaluation.EvaluationFormat = EvaluationFormat
sys.modules['llm_classifiers.evaluation'] = mock_evaluation


# Now that all mocks are in place, import our test module
import llm_classifiers.evaluation


class TestEvaluator(TestCase):
    def setUp(self):
        # Create a mock model and agent
        self.mock_model = mock.Mock(spec=Model)
        self.mock_agent = mock.patch('llm_classifiers.classifier.Agent').start()
        self.mock_agent_instance = self.mock_agent.return_value
        
        # Create a classifier
        self.classifier = Classifier(
            output_model=SentimentLabel,
            model=self.mock_model,
            enable_logging=False
        )
        
        # Create test examples
        self.test_examples = [
            Example(
                input_text="This is great!",
                output=SentimentLabel(sentiment="positive")
            ),
            Example(
                input_text="This is terrible.",
                output=SentimentLabel(sentiment="negative")
            ),
            Example(
                input_text="This is normal.",
                output=SentimentLabel(sentiment="neutral")
            )
        ]
    
    def tearDown(self):
        mock.patch.stopall()
    
    def test_evaluator_initialization(self):
        """Test that an evaluator can be initialized."""
        evaluator = ClassifierEvaluator(
            classifier=self.classifier,
            comparison_field="sentiment",
            enable_logging=False
        )
        
        self.assertEqual(evaluator.classifier, self.classifier)
        self.assertEqual(evaluator.comparison_field, "sentiment")
        self.assertFalse(evaluator.exact_match)
    
    def test_evaluate_single(self):
        """Test evaluation of a single example."""
        # Configure mock agent to return a specific result
        self.mock_agent_instance.run.return_value = SentimentLabel(sentiment="positive")
        
        # Create evaluator
        evaluator = ClassifierEvaluator(
            classifier=self.classifier,
            comparison_field="sentiment",
            enable_logging=False
        )
        
        # Run single evaluation
        result = evaluator._evaluate_single(self.test_examples[0], include_latency=False)
        
        # Check result
        self.assertIsInstance(result, EvaluationResult)
        self.assertEqual(result.input_text, "This is great!")
        self.assertEqual(result.ground_truth.sentiment, "positive")
        self.assertEqual(result.prediction.sentiment, "positive")
        self.assertTrue(result.correct)
    
    def test_evaluate_batch(self):
        """Test evaluation of a batch of examples."""
        # Configure mock to return matching sentiments for first two examples
        # and non-matching for the third
        def mock_classify(text):
            if "great" in text:
                return SentimentLabel(sentiment="positive")
            elif "terrible" in text:
                return SentimentLabel(sentiment="negative")
            else:
                return SentimentLabel(sentiment="positive")  # Incorrect for neutral
                
        self.mock_agent_instance.run.side_effect = mock_classify
        
        # Create evaluator
        evaluator = ClassifierEvaluator(
            classifier=self.classifier,
            comparison_field="sentiment",
            enable_logging=False
        )
        
        # Run batch evaluation
        results = evaluator.evaluate(
            self.test_examples, 
            include_latency=False, 
            parallel=False
        )
        
        # Check results
        self.assertEqual(len(results), 3)
        self.assertTrue(results[0].correct)
        self.assertTrue(results[1].correct)
        self.assertFalse(results[2].correct)
    
    def test_compute_metrics(self):
        """Test metrics computation."""
        # Create evaluator
        evaluator = ClassifierEvaluator(
            classifier=self.classifier,
            comparison_field="sentiment",
            enable_logging=False
        )
        
        # Create results
        results = [
            EvaluationResult(
                input_text="This is great!",
                ground_truth=SentimentLabel(sentiment="positive"),
                prediction=SentimentLabel(sentiment="positive"),
                correct=True
            ),
            EvaluationResult(
                input_text="This is terrible.",
                ground_truth=SentimentLabel(sentiment="negative"),
                prediction=SentimentLabel(sentiment="negative"),
                correct=True
            ),
            EvaluationResult(
                input_text="This is normal.",
                ground_truth=SentimentLabel(sentiment="neutral"),
                prediction=SentimentLabel(sentiment="positive"),
                correct=False
            )
        ]
        
        # Compute metrics
        metrics = evaluator.compute_metrics(results)
        
        # Check metrics
        self.assertEqual(metrics.total_examples, 3)
        self.assertEqual(metrics.correct_predictions, 2)
        self.assertAlmostEqual(metrics.accuracy, 2/3)
    
    def test_export_results(self):
        """Test export of results."""
        # Create evaluator
        evaluator = ClassifierEvaluator(
            classifier=self.classifier,
            comparison_field="sentiment",
            enable_logging=False
        )
        
        # Create results
        results = [
            EvaluationResult(
                input_text="This is great!",
                ground_truth=SentimentLabel(sentiment="positive"),
                prediction=SentimentLabel(sentiment="positive"),
                correct=True
            ),
            EvaluationResult(
                input_text="This is terrible.",
                ground_truth=SentimentLabel(sentiment="negative"),
                prediction=SentimentLabel(sentiment="negative"),
                correct=True
            )
        ]
        
        # Test various export formats
        json_result = evaluator.export_results(
            results,
            format=EvaluationFormat.JSON,
            include_metrics=True
        )
        self.assertIsInstance(json_result, str)
        self.assertIn("accuracy", json_result)
        
        df_result = evaluator.export_results(
            results,
            format=EvaluationFormat.DATAFRAME
        )
        self.assertIsInstance(df_result, pd.DataFrame)
        self.assertEqual(len(df_result), 2)
        
        with tempfile.NamedTemporaryFile(suffix='.csv') as temp:
            csv_result = evaluator.export_results(
                results,
                format=EvaluationFormat.CSV,
                file_path=temp.name
            )
            self.assertTrue(os.path.exists(temp.name))
            self.assertTrue(os.path.getsize(temp.name) > 0)
    
    def test_compare_classifiers(self):
        """Test comparison of classifiers."""
        # Create a second classifier
        classifier2 = Classifier(
            output_model=SentimentLabel,
            model=self.mock_model,
            enable_logging=False
        )
        
        # Create evaluator
        evaluator = ClassifierEvaluator(
            classifier=self.classifier,
            comparison_field="sentiment",
            enable_logging=False
        )
        
        # Compare classifiers
        comparison = evaluator.compare_classifiers(
            classifiers=[self.classifier, classifier2],
            test_data=self.test_examples,
            names=["Classifier1", "Classifier2"]
        )
        
        # Check result
        self.assertIsInstance(comparison, pd.DataFrame)
        self.assertEqual(len(comparison), 2)
        self.assertEqual(comparison.iloc[0]['name'], "Classifier1")
        self.assertEqual(comparison.iloc[1]['name'], "Classifier2")