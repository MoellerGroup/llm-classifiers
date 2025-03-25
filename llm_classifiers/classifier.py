from typing import Any, Dict, List, Optional, Type

import logfire
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIModel


class Example(logfire.LoggableModel):
    """Example for training a classifier."""

    input_text: str = Field(description="The input text to classify")
    output: BaseModel = Field(description="The expected classification output")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format required by pydantic-ai."""
        return {
            "input": self.input_text,
            "output": self.output
        }


class ClassificationResult(logfire.LoggableModel):
    """Result of a classification operation for logging."""

    input_text: str = Field(description="Input text that was classified")
    result: BaseModel = Field(description="Classification result")
    model_name: str = Field(description="Model used for classification")


class Classifier:
    """
    LLM-based classifier that enforces structured outputs.

    This class enables creating classifiers that validate outputs against
    a predefined Pydantic schema, optionally trained with examples.
    """

    def __init__(
        self,
        output_model: Type[BaseModel],
        model: Optional[Model] = None,
        system_prompt: Optional[str] = None,
        enable_logging: bool = True
    ):
        """
        Initialize a classifier.

        Args:
            output_model: The Pydantic model defining the classification
                schema.
            model: The LLM model to use (defaults to OpenAI if not provided).
            system_prompt: Optional system prompt to guide the classifier.
            enable_logging: Whether to enable logfire logging.
        """
        self.output_model = output_model
        self.model = model or OpenAIModel(model_name="gpt-3.5-turbo")
        self.system_prompt = system_prompt or "You are a classifier."
        self.agent = Agent(model=self.model)
        self._examples: List[Example] = []
        self._is_configured = False
        self._enable_logging = enable_logging

        if self._enable_logging:
            # Initialize logfire if not already initialized
            try:
                # Use a unique service name to avoid conflicts
                logfire.init(service_name="llm-classifier")
            except Exception:
                # Logfire might be already initialized, continue silently
                pass

            # Log initialization
            logfire.log(
                "Classifier initialized",
                output_model=output_model.__name__,
                model_name=getattr(self.model, "model_name", str(self.model)),
                system_prompt=self.system_prompt
            )

    @logfire.trace(skip_if=lambda self: not self._enable_logging)
    def add_example(self, input_text: str, output: BaseModel) -> None:
        """
        Add a training example to guide the classifier.

        Args:
            input_text: The input text to classify.
            output: The expected classification output.
        """
        example = Example(input_text=input_text, output=output)
        self._examples.append(example)
        self._is_configured = False  # Need to reconfigure

        if self._enable_logging:
            logfire.log("Example added",
                        example_text=input_text,
                        example_model=output.__class__.__name__)

    @logfire.trace(skip_if=lambda self: not self._enable_logging)
    def add_examples(self, examples: List[Example]) -> None:
        """
        Add multiple training examples to guide the classifier.

        Args:
            examples: List of Example objects.
        """
        self._examples.extend(examples)
        self._is_configured = False  # Need to reconfigure

        if self._enable_logging:
            logfire.log("Multiple examples added", count=len(examples))

    @logfire.trace(skip_if=lambda self: not self._enable_logging)
    def configure(self, system_prompt: Optional[str] = None) -> None:
        """
        Configure the agent with system prompt and examples.

        Args:
            system_prompt: Optional system prompt override.
        """
        prompt = system_prompt or self.system_prompt

        # Convert examples to the format expected by pydantic-ai
        examples_dict = [example.to_dict() for example in self._examples]

        self.agent.configure(
            system_prompt=prompt,
            examples=examples_dict,
            output_model=self.output_model
        )

        self._is_configured = True

        if self._enable_logging:
            logfire.log(
                "Classifier configured",
                examples_count=len(examples_dict),
                system_prompt=prompt
            )

    @logfire.trace(skip_if=lambda self: not self._enable_logging)
    def classify(self, text: str) -> BaseModel:
        """
        Classify the input text according to the schema.

        Args:
            text: The text to classify.

        Returns:
            Classification result as an instance of the output model.
        """
        if not self._is_configured:
            self.configure()

        if self._enable_logging:
            # Log start of classification
            logfire.log("Starting classification", input_text=text)

        # Execute the classification with timing if logging is enabled
        if self._enable_logging:
            with logfire.span("classification_execution"):
                result = self.agent.run(text)

            # Log the result using a structured model
            model_name = getattr(self.model, "model_name", str(self.model))
            classification_result = ClassificationResult(
                input_text=text,
                result=result,
                model_name=model_name
            )
            logfire.log("Classification completed", result=classification_result)
        else:
            result = self.agent.run(text)

        return result
