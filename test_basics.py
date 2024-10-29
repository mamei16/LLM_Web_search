import unittest

from llm_web_search import retrieve_from_duckduckgo, Generator
from retrieval import DocumentRetriever


class MyTestCase(unittest.TestCase):

    def setUp(self):
        self.document_retriever = DocumentRetriever(device="cpu", num_results=10, similarity_threshold=0.5, chunk_size=500,
                                                    ensemble_weighting=0.5, keyword_retriever="bm25",
                                                    chunking_method="character-based")

    def test_basic_search(self):
        gen = Generator(retrieve_from_duckduckgo("How much does a LLama weigh?",
                                                 self.document_retriever, instant_answers=True, max_results=5))
        status_messages = list(gen)
        self.assertEqual(status_messages[0], "Getting results from DuckDuckGo...")
        self.assertEqual(status_messages[1], "Downloading webpages...")
        self.assertEqual(status_messages[2], "Chunking page texts...")
        self.assertEqual(status_messages[3], "Retrieving relevant results...")

        search_result_dict = gen.retval
        self.assertEqual(len(search_result_dict), 5)

        for document in search_result_dict:
            self.assertIsNotNone(document.page_content)
            self.assertIsNotNone(document.metadata['source'])


if __name__ == '__main__':
    unittest.main()
