import unittest
import numpy as np
import pandas as pd

from text_embeddings import TextEmbedding


class TestTextEmbedding(unittest.TestCase):
    def setUp(self):
        # Initialize the TextEmbedding instance with a specific model checkpoint for testing
        self.llm = TextEmbedding()
        self.expected_shape = (768,)

    def test_output_type(self):
        test_text = "This is a test sentence."
        self.llm.set_text(test_text, n_augmentations=None)

        embedding = self.llm.get_embedding(output="pandas")
        self.assertTrue(isinstance(embedding, pd.DataFrame))

        embedding = self.llm.get_embedding(output="numpy")
        self.assertTrue(isinstance(embedding, np.ndarray))

        self.llm.set_text(test_text, n_augmentations=5)

        embedding = self.llm.get_embedding(output="pandas")
        self.assertTrue(isinstance(embedding, pd.DataFrame))

        embedding = self.llm.get_embedding(output="numpy")
        self.assertTrue(isinstance(embedding, np.ndarray))

    def test_set_text(self):
        # Test setting text and getting embedding
        test_text = "This is a test sentence."
        self.llm.set_text(test_text, n_augmentations=None)
        embedding = self.llm.get_embedding(output="numpy")

        # Check if the embedding is a numpy array
        self.assertTrue(isinstance(embedding, np.ndarray))

        # Check the shape of the embedding
        # Assuming the model used has an embedding dimension of 768
        self.assertEqual(embedding.shape, self.expected_shape)

    def test_set_text_list(self):
        # Test setting a list instead of a string, which should raise a TypeError
        test_text_list = ["This is a test sentence.", "Another sentence."]
        with self.assertRaises(TypeError):
            self.llm.set_text(test_text_list, n_augmentations=None)

    def test_long_text(self):
        # Test setting a long text that exceeds max_length
        long_text = """Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vestibulum volutpat elit ac eros 
        sollicitudin, in fermentum elit pharetra. Sed ut malesuada nisi, id faucibus purus. Maecenas sit amet odio 
        nec quam facilisis sollicitudin. Curabitur malesuada, ex in tristique convallis, velit quam pellentesque 
        justo, non feugiat nulla justo ac sapien. Vivamus vel sem in nisl dapibus accumsan. Fusce quis scelerisque 
        lorem. Duis nec tristique velit. Fusce posuere eros in quam blandit auctor. Nulla eu risus quis augue gravida 
        gravida. Sed vitae magna eget risus auctor rhoncus. Integer euismod ex a malesuada condimentum. Vivamus eu 
        augue sit amet velit fermentum tristique. Aenean auctor tincidunt turpis id auctor. Nullam in justo vel orci 
        cursus tempus. Suspendisse vel laoreet mi. Fusce elementum libero ut arcu posuere, a semper nulla vestibulum. 
        Sed vel odio in eros tincidunt elementum ac in nisi. Maecenas id tristique arcu. Suspendisse ut turpis nec 
        est efficitur blandit nec eget libero. Donec dapibus, ex nec sagittis elementum, orci odio tincidunt justo, 
        sit amet vestibulum nisi risus nec augue. Cras sed ex in dui bibendum interdum eget sit amet mi. Vivamus 
        lacinia efficitur bibendum. Sed id mauris sit amet massa cursus imperdiet a nec nisi. Pellentesque tincidunt, 
        erat non dapibus placerat, enim orci vulputate justo, in tempus urna quam nec erat. In suscipit bibendum odio 
        in mattis. Sed tincidunt purus et bibendum varius. Nullam sit amet dolor id enim eleifend elementum."""

        self.llm.set_text(long_text, n_augmentations=None)
        embedding = self.llm.get_embedding(output="numpy")

        # Check if the embedding is still 1D
        self.assertTrue(isinstance(embedding, np.ndarray))
        self.assertEqual(embedding.ndim, 1)
        self.assertEqual(embedding.shape, self.expected_shape)

    def test_augmentations(self):
        # Test setting text with augmentations
        test_text = """1This is a test sentence. 1The second sentence is here. 1And a third one.§ 2Another sentence. 
        2And another one. 2And another one. 2This is the last sentence.§ 3This is a test sentence. 3The second sentence.
        """
        self.llm.set_text(test_text, n_augmentations=5)

        embedding = self.llm.get_embedding(output="numpy")
        self.assertTrue(isinstance(embedding, np.ndarray))
        # Check the shape of the embedding
        # Assuming the model used has an embedding dimension of 768
        # The embedding should be a 2D array with shape (768, 6)
        # 6 = 1 original sentence + 5 augmentations
        self.assertEqual(embedding.shape, (768, 6))
        for i in range(1, 6):
            self.assertFalse(np.array_equal(embedding[:, 0], embedding[:, i]))

    def test_augmentation_determinism(self):
        test_text = "This is a test sentence."
        self.llm.set_text(test_text, n_augmentations=5)
        embedding = self.llm.get_embedding(output="numpy")

        # Check if the embeddings for one sentence are the same for every augmentation
        for i in range(1, 6):
            self.assertTrue(np.array_equal(embedding[:, 0], embedding[:, i]))

    def tearDown(self):
        # Clean up any resources if needed
        pass


if __name__ == "__main__":
    unittest.main()
