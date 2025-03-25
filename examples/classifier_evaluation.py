from typing import List, Literal

import logfire
import pandas as pd
from pydantic import BaseModel, Field
from pydantic_ai.models.openai import OpenAIModel

from llm_classifiers.classifier import Classifier, Example
from llm_classifiers.evaluation import ClassifierEvaluator, EvaluationFormat


class SentimentClassification(BaseModel):
    """Schema for sentiment classification."""
    
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="The sentiment of the text"
    )
    confidence: float = Field(
        description="Confidence score between 0 and 1",
        ge=0.0,
        le=1.0
    )


def create_test_dataset() -> List[Example]:
    """Create a dataset for testing the classifier."""
    return [
        Example(
            input_text="This product is amazing! I can't believe how well it works.",
            output=SentimentClassification(sentiment="positive", confidence=0.95)
        ),
        Example(
            input_text="The service was terrible. I'll never shop here again.",
            output=SentimentClassification(sentiment="negative", confidence=0.9)
        ),
        Example(
            input_text="The package arrived on time. Standard delivery as expected.",
            output=SentimentClassification(sentiment="neutral", confidence=0.8)
        ),
        Example(
            input_text="I love how easy this app is to use. Best purchase ever!",
            output=SentimentClassification(sentiment="positive", confidence=0.9)
        ),
        Example(
            input_text="Very disappointed with the quality. It broke after one use.",
            output=SentimentClassification(sentiment="negative", confidence=0.9)
        ),
        Example(
            input_text="The product functions as described. Nothing special.",
            output=SentimentClassification(sentiment="neutral", confidence=0.7)
        ),
        Example(
            input_text="Worst customer service I've ever experienced.",
            output=SentimentClassification(sentiment="negative", confidence=0.95)
        ),
        Example(
            input_text="I received the order yesterday. It's the standard model.",
            output=SentimentClassification(sentiment="neutral", confidence=0.8)
        ),
        Example(
            input_text="Excellent product and fast shipping. Very happy!",
            output=SentimentClassification(sentiment="positive", confidence=0.9)
        ),
        Example(
            input_text="Not what I expected. Wouldn't recommend.",
            output=SentimentClassification(sentiment="negative", confidence=0.7)
        )
    ]


def main():
    """Run a classifier evaluation example."""
    # Initialize logfire for structured logging
    logfire.init(
        service_name="classifier-evaluation-example",
        level="INFO",
        handler_options={
            "console": {"enabled": True},
            "file": {
                "enabled": True,
                "filename": "evaluation.log"
            }
        }
    )
    
    logfire.log("Starting classifier evaluation example")
    
    # Create two different classifiers with different system prompts
    model = OpenAIModel(model_name="gpt-3.5-turbo")
    
    # First classifier with a basic prompt
    classifier1 = Classifier(
        output_model=SentimentClassification,
        model=model,
        system_prompt="You are a sentiment analyzer. Classify text as positive, negative, or neutral.",
        enable_logging=True
    )
    
    # Second classifier with a more detailed prompt
    classifier2 = Classifier(
        output_model=SentimentClassification,
        model=model,
        system_prompt="""You are a sentiment analyzer. Classify text as positive, negative, or neutral.
        
Guidelines:
- Positive: Contains explicit praise, enthusiasm, or satisfaction
- Negative: Contains criticism, disappointment, or dissatisfaction
- Neutral: Contains factual statements without strong emotions
        
Be careful with sarcasm and implicit sentiment. Focus on the overall tone.""",
        enable_logging=True
    )
    
    # Create examples for training (different from evaluation set)
    training_examples = [
        Example(
            input_text="I absolutely love this product!",
            output=SentimentClassification(sentiment="positive", confidence=0.95)
        ),
        Example(
            input_text="This was a complete waste of money.",
            output=SentimentClassification(sentiment="negative", confidence=0.9)
        ),
        Example(
            input_text="The item arrived yesterday afternoon.",
            output=SentimentClassification(sentiment="neutral", confidence=0.85)
        )
    ]
    
    # Add the same examples to both classifiers
    for example in training_examples:
        classifier1.add_example(example.input_text, example.output)
        classifier2.add_example(example.input_text, example.output)
    
    # Configure the classifiers
    classifier1.configure()
    classifier2.configure()
    
    # Create test dataset
    test_data = create_test_dataset()
    
    # Evaluate single classifier
    print("Evaluating Classifier 1...")
    evaluator = ClassifierEvaluator(
        classifier=classifier1,
        comparison_field="sentiment",  # Only compare the sentiment field for correctness
        exact_match=False,
        max_workers=4
    )
    
    # Run evaluation
    results = evaluator.evaluate(test_data, include_latency=True, parallel=True)
    
    # Compute and print metrics
    metrics = evaluator.compute_metrics(results)
    print(metrics)
    print("\n" + "="*50 + "\n")
    
    # Export results in different formats
    print("Exporting results to files...")
    
    # Save as JSON
    evaluator.export_results(
        results, 
        format=EvaluationFormat.JSON,
        file_path="classifier1_results.json",
        include_metrics=True
    )
    
    # Save as CSV
    evaluator.export_results(
        results, 
        format=EvaluationFormat.CSV,
        file_path="classifier1_results.csv"
    )
    
    # Get as DataFrame
    df = evaluator.export_results(
        results, 
        format=EvaluationFormat.DATAFRAME
    )
    print(f"Result DataFrame Shape: {df.shape}")
    
    # Compare two classifiers
    print("\nComparing classifiers with different prompts...")
    comparison = evaluator.compare_classifiers(
        classifiers=[classifier1, classifier2],
        test_data=test_data,
        names=["Basic Prompt", "Detailed Prompt"],
        include_latency=True
    )
    
    # Print comparison
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print("\nClassifier Comparison:")
    print(comparison[["name", "accuracy", "average_latency_ms"]])
    
    # Also save comparison to CSV
    comparison.to_csv("classifier_comparison.csv", index=False)
    
    logfire.log("Classifier evaluation example completed")


if __name__ == "__main__":
    main()