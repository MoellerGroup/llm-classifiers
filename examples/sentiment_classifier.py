from typing import Literal

import logfire
from pydantic import BaseModel, Field
from pydantic_ai.models.openai import OpenAIModel

from llm_classifiers.classifier import Classifier, Example


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
    reasoning: str = Field(
        description="Explanation of why this sentiment was chosen"
    )


def main():
    """Run a simple sentiment classification example."""
    # Initialize logfire for structured logging
    logfire.init(
        service_name="sentiment-classifier-example",
        level="INFO",
        handler_options={
            "console": {"enabled": True},
            "file": {
                "enabled": True,
                "filename": "sentiment_classifier.log"
            }
        }
    )
    
    logfire.log("Starting sentiment classification example")
    
    # Initialize the classifier with our schema
    model = OpenAIModel(model_name="gpt-3.5-turbo")
    classifier = Classifier(
        output_model=SentimentClassification,
        model=model,
        system_prompt="You are a sentiment analyzer that classifies text as positive, negative, or neutral.",
        enable_logging=True  # Enable detailed logfire logging
    )
    
    # Add some examples to guide the classifications
    with logfire.span("adding_examples"):
        classifier.add_example(
            "I absolutely love this product! It's been a game changer for me.",
            SentimentClassification(
                sentiment="positive",
                confidence=0.95,
                reasoning="Uses strong positive language ('love', 'game changer') with exclamation marks."
            )
        )
        
        classifier.add_example(
            "This service is terrible. I've been waiting for hours and no one has helped me.",
            SentimentClassification(
                sentiment="negative",
                confidence=0.9,
                reasoning="Contains explicit negative language ('terrible') and describes a negative experience."
            )
        )
        
        classifier.add_example(
            "The product arrived on time. It works as described.",
            SentimentClassification(
                sentiment="neutral",
                confidence=0.8,
                reasoning="States facts without strong positive or negative language."
            )
        )
    
    # Configure the classifier with our examples
    classifier.configure()
    
    # Test with some new texts
    texts = [
        "I'm really impressed with the quality and speed of delivery.",
        "This is the worst purchase I've ever made. Don't waste your money.",
        "The item was delivered yesterday. Standard packaging."
    ]
    
    # Run the classifier on each text
    for text in texts:
        with logfire.span(f"classifying_text"):
            result = classifier.classify(text)
            
            print(f"Text: {text}")
            print(f"Sentiment: {result.sentiment}")
            print(f"Confidence: {result.confidence}")
            print(f"Reasoning: {result.reasoning}")
            print("-" * 50)
    
    logfire.log("Sentiment classification example completed")


if __name__ == "__main__":
    main()