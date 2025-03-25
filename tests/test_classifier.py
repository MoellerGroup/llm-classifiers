from unittest import TestCase, mock
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai.models import Model

from llm_classifiers.classifier import Classifier, Example


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
        # Create a mock model
        mock_model = mock.Mock(spec=Model)
        classifier = Classifier(output_model=SentimentResult, model=mock_model)
        self.assertEqual(classifier.output_model, SentimentResult)
        self.assertEqual(classifier._examples, [])
    
    def test_add_example(self):
        """Test that examples can be added to a classifier."""
        mock_model = mock.Mock(spec=Model)
        classifier = Classifier(output_model=SentimentResult, model=mock_model)
        example_output = SentimentResult(sentiment="positive", confidence=0.9)
        
        classifier.add_example("Great product!", example_output)
        
        self.assertEqual(len(classifier._examples), 1)
        self.assertEqual(classifier._examples[0].input_text, "Great product!")
        self.assertEqual(classifier._examples[0].output, example_output)
    
    def test_add_examples(self):
        """Test that multiple examples can be added at once."""
        mock_model = mock.Mock(spec=Model)
        classifier = Classifier(output_model=SentimentResult, model=mock_model)
        
        examples = [
            Example("Great product!", SentimentResult(sentiment="positive", confidence=0.9)),
            Example("Terrible service", SentimentResult(sentiment="negative", confidence=0.8))
        ]
        
        classifier.add_examples(examples)
        
        self.assertEqual(len(classifier._examples), 2)
    
    def test_example_to_dict(self):
        """Test that example conversion to dict works correctly."""
        output = SentimentResult(sentiment="positive", confidence=0.9)
        example = Example("Great product!", output)
        
        result = example.to_dict()
        
        self.assertEqual(result["input"], "Great product!")
        self.assertEqual(result["output"], output)
    
    @mock.patch('llm_classifiers.classifier.Agent')
    def test_configure(self, mock_agent):
        """Test that agent is configured correctly."""
        # Setup
        mock_model = mock.Mock(spec=Model)
        classifier = Classifier(output_model=SentimentResult, model=mock_model)
        example_output = SentimentResult(sentiment="positive", confidence=0.9)
        classifier.add_example("Great product!", example_output)
        
        # Execute
        classifier.configure("You are a sentiment analyzer")
        
        # Assert
        mock_agent_instance = mock_agent.return_value
        mock_agent_instance.configure.assert_called_once()
        # Verify the arguments passed to configure
        args, kwargs = mock_agent_instance.configure.call_args
        self.assertEqual(kwargs["system_prompt"], "You are a sentiment analyzer")
        self.assertEqual(kwargs["output_model"], SentimentResult)
        self.assertEqual(len(kwargs["examples"]), 1)
        
    @mock.patch('llm_classifiers.classifier.Agent')
    def test_classify(self, mock_agent):
        """Test that classification works correctly."""
        # Setup
        mock_agent_instance = mock_agent.return_value
        expected_result = SentimentResult(sentiment="positive", confidence=0.9)
        mock_agent_instance.run.return_value = expected_result
        
        # Execute
        mock_model = mock.Mock(spec=Model)
        classifier = Classifier(output_model=SentimentResult, model=mock_model)
        result = classifier.classify("Great product!")
        
        # Assert
        self.assertEqual(result, expected_result)
        mock_agent_instance.run.assert_called_with("Great product!")