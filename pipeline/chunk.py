import json
from pathlib import Path

class Chunker:
    def __init__(self, chunk_size, chunk_stride, tokenizer):
        self.chunk_size   = chunk_size
        self.chunk_stride = chunk_stride
        self.tokenizer    = tokenizer

    def chunk_text(self, text_path: Path, chunks_path: Path):
        text   = text_path.read_text(encoding='utf-8')
        enc    = self.tokenizer(text)
        tokens = enc.input_ids

        chunks = []
        for i, start_tok in enumerate(range(0, len(tokens), self.chunk_stride)):
            slice_ids = tokens[start_tok : start_tok + self.chunk_size]
            chunk_text = self.tokenizer.decode(slice_ids,
                                               clean_up_tokenization_spaces=False,
                                               skip_special_tokens=True)
            chunks.append({
                "chunk_id": f"{i:06d}",
                "text": chunk_text
            })

            # Break if we've reached the end of the document
            if start_tok + self.chunk_size >= len(tokens):
                break

        # Save to file
        with open(chunks_path, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")