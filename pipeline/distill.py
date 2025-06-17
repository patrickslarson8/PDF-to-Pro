import json
import logging
from pathlib import Path
from openai import OpenAI
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor, as_completed


class QA(BaseModel):
    """
    Pydantic model for the structure of the response to receive from OpenAI.
    """
    question: str
    answer:   str


class Getter:
    """
    Class to load chunked data and retrieve training data in question/answer format.

    Attributes:
        api_key: OpenAI API key.
        model: OpenAI model to use.
        prompt: Prompt to send with chunks.
    """
    def __init__(self, api_key: str, model: str, prompt: str):
        self.api_key = api_key
        self.model = model
        self.prompt = prompt

    def load_chunks(self, file: str) -> list:
        """
        Parses json saved chunks into list.
        :param file: Location of JSON.
        :return: List of json contents.
        """
        chunks = []
        with Path(file).open() as f:
            for line in f:
                chunks.append(json.loads(line))
        return chunks

    def _call_openai(self, chunk, model: str, system_prompt: str) -> dict | None:
        """
        Creates a client and prompt to call OpenAI and returns the response.
        :param chunk: Section of text to use as context.
        :param model: Model from OpenAI to use.
        :param system_prompt: Prompt to send with chunks.
        :return: The result of the OpenAI request parsed according to the Pydantic model.
        """
        try:
            client = OpenAI(api_key=self.api_key)
            passage = chunk["text"]
            resp = client.responses.parse(
                model=model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": passage}
                ],
                text_format=QA
            )
            q = resp.output_parsed.question
            a = resp.output_parsed.answer
            return {
                "chunk_id": chunk.get("chunk_id"),
                "context": passage,
                "question": q,
                "answer": a
            }
        except Exception as e:
            logging.warning(f"Error returned from OpenAI for chunk {chunk.get('chunk_id', '')}: {e}")
            return None

    def get_qa_pairs(self,
                     chunked_dir: str,
                     output_path: str,
                     max_calls: int,
                     num_workers: int):
        """
        The main function which concurrently spawns workers to call
        OpenAPI and save the responses.
        :param chunked_dir: Directory to load context chunks from.
        :param output_path: Location to save responses.
        :param max_calls: Maximum number of requests to send.
        :param num_workers: Number of workers to use.
        :return: None
        """
        chunks = self.load_chunks(chunked_dir)
        responses = []

        with open(output_path, "w", encoding="utf-8") as outfile:
            while len(responses) < max_calls and len(responses) < len(2 * chunks):
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = [
                        executor.submit(self._call_openai,
                                        chunk,
                                        self.model,
                                        self.prompt)
                        for chunk in chunks
                    ]
                    for i, f in enumerate(as_completed(futures)):
                        result = f.result()
                        if result:
                            outfile.write(json.dumps(result, ensure_ascii=False) + "\n")
                            outfile.flush()
                            responses.append(result)