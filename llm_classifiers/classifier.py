from typing import Any, Dict, List, Optional, Type, TypeVar, Generic

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIModel


T = TypeVar('T', bound=BaseModel)


class Example(Generic[T]):
    """Example for training a classifier."""

    def __init__(self, input_text: str, output: T):
        """
        Initialize an example.

        Args:
            input_text: The input text to classify.
            output: The expected output classification.
        """
        self.input_text = input_text
        self.output = output

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format required by pydantic-ai."""
        return {
            "input": self.input_text,
            "output": self.output
        }


class Classifier(Generic[T]):
    """
    LLM-based classifier that enforces structured outputs.

    This class enables creating classifiers that validate outputs against
    a predefined Pydantic schema, optionally trained with examples.
    """

    def __init__(
        self,
        output_model: Type[T],
        model: Optional[Model] = None,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize a classifier.

        Args:
            output_model: The Pydantic model defining the classification
                schema.
            model: The LLM model to use (defaults to OpenAI if not provided).
            system_prompt: Optional system prompt to guide the classifier.
        """
        self.output_model = output_model
        self.model = model or OpenAIModel(model_name="gpt-3.5-turbo")
        self.system_prompt = system_prompt or "You are a classifier."
        self.agent = Agent(model=self.model)
        self._examples: List[Example[T]] = []
        self._is_configured = False

    def add_example(self, input_text: str, output: T) -> "Classifier[T]":
        """
        Add a training example to guide the classifier.

        Args:
            input_text: The input text to classify.
            output: The expected classification output.

        Returns:
            The classifier instance for method chaining.
        """
        example = Example(input_text, output)
        self._examples.append(example)
        self._is_configured = False  # Need to reconfigure
        return self

    def add_examples(self, examples: List[Example[T]]) -> "Classifier[T]":
        """
        Add multiple training examples to guide the classifier.

        Args:
            examples: List of Example objects.

        Returns:
            The classifier instance for method chaining.
        """
        self._examples.extend(examples)
        self._is_configured = False  # Need to reconfigure
        return self

    def configure(
        self, system_prompt: Optional[str] = None
    ) -> "Classifier[T]":
        """
        Configure the agent with system prompt and examples.

        Args:
            system_prompt: Optional system prompt override.

        Returns:
            The classifier instance for method chaining.
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
        return self

    def classify(self, text: str) -> T:
        """
        Classify the input text according to the schema.

        Args:
            text: The text to classify.

        Returns:
            Classification result as an instance of the output model.
        """
        if not self._is_configured:
            self.configure()

        result = self.agent.run(text)
        return result
