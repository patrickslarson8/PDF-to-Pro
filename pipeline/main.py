import os
import shutil

from dotenv import load_dotenv
from pathlib import Path

from model import LLM
from vec_db import VectorDB


def get_lora_config():
    config = {
        "LORA_MODULES": os.environ["LORA_MODULES"].split(','),
        "LORA_ALPHA":   int(os.environ["LORA_ALPHA"]),
        "LORA_R":       int(os.environ["LORA_R"]),
        "LORA_DROPOUT": float(os.environ["LORA_DROPOUT"]),
        "LORA_BIAS":    os.environ["LORA_BIAS"],
    }
    return config

def _get_gguf_path(search_root: Path, specific_model: str):
    for path in search_root.rglob(specific_model):
        return path
    return None

if __name__ == "__main__":

    load_dotenv()

    DOCUMENT       = Path(os.environ["DOCUMENT"])
    BASE_MODEL     = os.environ["BASE_MODEL"]
    SPECIFIC_MODEL = os.environ["SPECIFIC_MODEL"]
    CACHE_DIR      = Path(os.environ.get("CACHE_DIR"))

    DATA_DIR     = Path(DOCUMENT.stem) / BASE_MODEL / "data"
    MODEL_DIR    = Path(DOCUMENT.stem) / BASE_MODEL / "lora"
    RAW_TEXT     = DATA_DIR            / "raw"      / "extracted.txt"
    CHUNKED_DATA = DATA_DIR            / "chunked"  / "chunked.jsonl"
    QA_DATA      = DATA_DIR            / "qa"       / "qa_pairs.jsonl"

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    if not RAW_TEXT.exists():
        from pipeline.ingest import Ingestor
        RAW_TEXT.parent.mkdir(parents=True, exist_ok=True)
        try:
            ingestor = Ingestor()
            ingestor.save_text(DOCUMENT, RAW_TEXT)
        except Exception as e:
            RAW_TEXT.unlink(missing_ok=True)
            raise e

    model = None
    needs_finetuning = False
    if not MODEL_DIR.exists():
        try:
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            needs_finetuning = True
            if not CACHE_DIR.exists():
                CACHE_DIR.mkdir(parents=True, exist_ok=True)

            from huggingface_hub import login
            HF_API_KEY = os.environ["HF_API_KEY"]
            login(HF_API_KEY)

            lora_conf = get_lora_config()
            model = LLM(BASE_MODEL,
                        SPECIFIC_MODEL,
                        MODEL_DIR,
                        CACHE_DIR,
                        lora_conf)
            model.save_model(MODEL_DIR.as_posix())
        except Exception as e:
            shutil.rmtree(MODEL_DIR.as_posix())
            raise e
    else:
        model = LLM(BASE_MODEL,
                    SPECIFIC_MODEL,
                    MODEL_DIR,
                    CACHE_DIR)

    if not CHUNKED_DATA.exists():
        needs_finetuning = True
        try:
            from pipeline.chunk import Chunker
            CHUNKED_DATA.parent.mkdir(parents=True, exist_ok=True)
            chunker = Chunker(int(os.environ["CHUNK_SIZE"]),
                              int(os.environ["CHUNK_STRIDE"]),
                              model.tokenizer)
            chunker.chunk_text(RAW_TEXT, CHUNKED_DATA)
        except Exception as e:
            CHUNKED_DATA.unlink(missing_ok=True)
            raise e

    if not QA_DATA.exists():
        needs_finetuning = True
        try:
            from pipeline.distill import Getter
            QA_DATA.parent.mkdir(parents=True, exist_ok=True)
            getter = Getter(os.environ["OAI_API_KEY"],
                            os.environ["OPENAI_MODEL"],
                            os.environ["QA_PROMPT"])
            getter.get_qa_pairs(CHUNKED_DATA,
                                QA_DATA,
                                int(os.environ["MAX_API_CALLS"]),
                                int(os.environ["NUM_WORKERS"]))
        except Exception as e:
            QA_DATA.unlink(missing_ok=True)
            raise e

    needs_finetuning = True
    if needs_finetuning:
        ds_train, ds_test = model.load_dataset(QA_DATA.as_posix(),
                                               os.environ["SYS_PROMPT"],
                                               test_portion = 0.1)
        model.train(ds_train, ds_test)

    db = VectorDB(DOCUMENT.stem, CHUNKED_DATA)

    while True:
        query = input(f"How can I help with {DOCUMENT.stem}?\n")
        if query == 'q':
            print("See you later!")
            break

        print("Good question, let me see...")
        contexts = db.retrieve_relevant_chunks(query)
        response = model.inference(os.environ["SYS_PROMPT"], contexts, query)
        print(response)
