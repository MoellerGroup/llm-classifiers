from unittest import TestCase

from llm_classifiers.dummy_method import add


class TestDummyMethod(TestCase):
    def test_add(self):
        self.assertEqual(add(1, 2), 3)
