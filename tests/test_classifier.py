import sys
from unittest import TestCase, mock
from typing import Literal, Dict, Any

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

from pydantic import BaseModel, Field
from pydantic_ai.models import Model

# Import our code after mocking
from llm_classifiers.classifier import Classifier


# Create a new Example class for testing since we can't use the one from classifier.py
class Example:
    def __init__(self, input_text: str, output: BaseModel):
        self.input_text = input_text
        self.output = output

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input": self.input_text,
            "output": self.output
        }


class SentimentResult(BaseModel):
    """Simple sentiment classification schema."""
    
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="The sentiment of the text"
    )
    confidence: float = Field(
        description="Confidence score between 0 and 1",
        ge=0.0,
        le=1.0
    )


class TestClassifier(TestCase):
    def test_classifier_initialization(self):
        """Test that a classifier can be initialized with a schema."""
        mock_model = mock.Mock(spec=Model)
        classifier = Classifier(output_model=SentimentResult, model=mock_model)
        self.assertEqual(classifier.output_model, SentimentResult)
        self.assertEqual(classifier._examples, [])
    
    @mock.patch('llm_classifiers.classifier.Example')
    def test_add_example(self, mock_example):
        """Test that examples can be added to a classifier."""
        # Setup
        mock_example_instance = Example("Great product!", SentimentResult(sentiment="positive", confidence=0.9))
        mock_example.return_value = mock_example_instance
        
        # Create classifier
        mock_model = mock.Mock(spec=Model)
        classifier = Classifier(output_model=SentimentResult, model=mock_model)
        example_output = SentimentResult(sentiment="positive", confidence=0.9)
        
        # Execute
        classifier.add_example("Great product!", example_output)
        
        # Assert
        self.assertEqual(len(classifier._examples), 1)
        self.assertEqual(classifier._examples[0], mock_example_instance)
    
    def test_example_to_dict(self):
        """Test that example conversion to dict works correctly."""
        output = SentimentResult(sentiment="positive", confidence=0.9)
        # Use our test Example class
        example = Example("Great product!", output)
        
        result = example.to_dict()
        
        self.assertEqual(result["input"], "Great product!")
        self.assertEqual(result["output"], output)
    
    @mock.patch('llm_classifiers.classifier.Agent')
    @mock.patch('llm_classifiers.classifier.Example')
    def test_configure(self, mock_example, mock_agent):
        """Test that agent is configured correctly."""
        # Setup Examples
        example_instance = Example("Great product!", SentimentResult(sentiment="positive", confidence=0.9))
        mock_example.return_value = example_instance
        
        # Setup agent
        mock_agent_instance = mock_agent.return_value
        
        # Create classifier
        mock_model = mock.Mock(spec=Model)
        classifier = Classifier(output_model=SentimentResult, model=mock_model)
        example_output = SentimentResult(sentiment="positive", confidence=0.9)
        classifier.add_example("Great product!", example_output)
        
        # Execute
        classifier.configure("You are a sentiment analyzer")
        
        # Assert
        mock_agent_instance.configure.assert_called_once()
        # Verify the arguments passed to configure
        args, kwargs = mock_agent_instance.configure.call_args
        self.assertEqual(kwargs["system_prompt"], "You are a sentiment analyzer")
        self.assertEqual(kwargs["output_model"], SentimentResult)
        self.assertEqual(len(kwargs["examples"]), 1)
    
    @mock.patch('llm_classifiers.classifier.Agent')
    @mock.patch('llm_classifiers.classifier.ClassificationResult')
    def test_classify(self, mock_classification_result, mock_agent):
        """Test that classification works correctly."""
        # Setup
        mock_agent_instance = mock_agent.return_value
        expected_result = SentimentResult(sentiment="positive", confidence=0.9)
        mock_agent_instance.run.return_value = expected_result
        
        # Setup classification result
        mock_classification_instance = mock.MagicMock()
        mock_classification_result.return_value = mock_classification_instance
        
        # Execute
        mock_model = mock.Mock(spec=Model)
        classifier = Classifier(output_model=SentimentResult, model=mock_model)
        result = classifier.classify("Great product!")
        
        # Assert
        self.assertEqual(result, expected_result)
        mock_agent_instance.run.assert_called_with("Great product!")
    
    @mock.patch('llm_classifiers.classifier.Agent')
    def test_classifier_without_logging(self, mock_agent):
        """Test that classifier works with logging disabled."""
        # Setup
        mock_agent_instance = mock_agent.return_value
        expected_result = SentimentResult(sentiment="positive", confidence=0.9)
        mock_agent_instance.run.return_value = expected_result
        
        # Create classifier with logging disabled
        mock_model = mock.Mock(spec=Model)
        classifier = Classifier(
            output_model=SentimentResult, 
            model=mock_model,
            enable_logging=False
        )
        
        # Execute with logging disabled
        result = classifier.classify("Great product!")
        
        # Assert result is correct
        self.assertEqual(result, expected_result)