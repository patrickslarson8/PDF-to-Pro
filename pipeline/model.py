from huggingface_hub import hf_hub_download
from torch import bfloat16
from trl import DataCollatorForCompletionOnlyLM
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    EarlyStoppingCallback, Seq2SeqTrainer, Seq2SeqTrainingArguments)
from peft import LoraConfig, get_peft_model, PeftModel

sys_role     = "system"
usr_role     = "user"
bot_role     = "assistant"
context_beg  = "<CONTEXT>"
context_end  = "</CONTEXT>"
bos_tok      = "<|begin_of_text|>"
eot_id_tok   = "<|eot_id|>"
start_hd_tok = "<|start_header_id|>"
end_hd_tok   = "<|end_header_id|>"
eot_tok      = "<|end_of_text|>"

class LLM:
    """
    Lora-adapted class that can be used with any Llama3.2 model from HuggingFace.

    Attributes:
        base_model_str: specifies which model to load
        specific_model_str: specifies which model to load
        model_dir: location to save adapters and tokenizer
        cache_dir: location HuggingFace will save the base model
        lora_params: dictionary of LoRA parameters
        tokenizer: tokenizer used for training and inference
        base_model: stores the base (non-adapted) model
        lora_model: stores the specific (adapted) model
        meteor: instance of huggingface meteor
        max_len: maximum length of tokens when training
    """
    def __init__(self,
                 base_model,
                 specific_model,
                 model_dir,
                 cache_dir,
                 lora_params=None):
        """
        Initializes and attempts to load the model.
        """
        self.base_model_str     = base_model
        self.specific_model_str = specific_model
        self.model_dir          = model_dir
        self.cache_dir          = cache_dir
        self.lora_params        = lora_params
        self.tokenizer          = None
        self.base_model         = None
        self.lora_model         = None
        self.meteor             = None
        self.max_len            = 750

        self._build_model(lora_params)

    def _build_model(self, lora_params):
        """
        Loads the base model and LoRA adapters, or downloads and creates them if they don't exist.
        :param lora_params: dictionary of LoRA parameters
        :return: None.
        """
        hf_hub_download(
            repo_id   = self.base_model_str,
            filename  = self.specific_model_str,
            cache_dir = self.cache_dir
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_str,
                                                       cache_dir=self.cache_dir,
                                                       gguf_file=self.specific_model_str,
                                                       use_fast=True)
        self.tokenizer.pad_token = "<|finetune_right_pad_id|>"

        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_str,
            cache_dir=self.cache_dir,
            gguf_file=self.specific_model_str,
            device_map="auto",
            torch_dtype=bfloat16)
        self.base_model.gradient_checkpointing_enable()
        try:
            self.lora_model = PeftModel.from_pretrained(self.base_model,
                                                        self.model_dir,
                                                        is_trainable=True)

        except ValueError as e:
            lora_cfg = LoraConfig(
                r              = lora_params['LORA_R'],
                lora_alpha     = lora_params['LORA_ALPHA'],
                target_modules = lora_params['LORA_MODULES'],
                lora_dropout   = lora_params['LORA_DROPOUT'],
                bias           = lora_params['LORA_BIAS'],
                task_type      = "CAUSAL_LM")

            self.lora_model = get_peft_model(self.base_model, lora_cfg)


    def save_model(self, model_dir):
        """
        Saves the model with tokenizer.
        :param model_dir: Where to save.
        :return: None.
        """
        self.lora_model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)

    def build_prompt(self, sys: str, contexts: str | list, usr: str, ans: str=None) -> str:
        """
        Creates the prompt in a repeatable way.
        :param sys: System prompt.
        :param contexts: String or list of strings used as context for the model.
        :param usr: User question.
        :param ans: Optional "best" answer for building training prompts.
        :return: The complete prompt.
        """
        if isinstance(contexts, str):
            combined_context = contexts
        else:
            combined_context = "\n\n".join(contexts)
        prompt  = f"{start_hd_tok}{sys_role}{end_hd_tok}{sys}{eot_id_tok}"
        prompt += f"{start_hd_tok}{usr_role}{end_hd_tok}{context_beg}{combined_context}{context_end}{usr}{eot_id_tok}"
        prompt += f"{start_hd_tok}{bot_role}{end_hd_tok}"

        if ans is not None:
            prompt += f"{ans}{eot_id_tok}{eot_tok}"

        return prompt

    def row_to_prompt(self, row: list, sys_prompt: str):
        """
        Creates training prompt from a dataset row.
        :param row: Dataset row.
        :param sys_prompt: System prompt.
        :return: Complete training prompt.
        """
        txt = self.build_prompt(
            sys_prompt,
            row["context"],
            row["question"],
            ans=row["answer"]
        )
        return {"text": txt}

    def tokenize(self, batch: list) -> list:
        """
        Creates tokenized prompt.
        :param batch: List of prompts.
        :return: List of tokenized prompts.
        """
        return self.tokenizer(
            batch["text"],
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len)

    def load_dataset(self, path: str, sys_prompt: str, test_portion: float=0.2) -> tuple[Dataset, Dataset]:
        """
        Loads, processes, and tokenizes the dataset.
        :param path: local path to dataset.
        :param sys_prompt: system prompt.
        :param test_portion: portion of dataset for testing.
        :return: The dataset ready for inference. Returns training and testing data with split or all as training
        data if split = 0.0.
        """
        raw_ds    = load_dataset("json", data_files=path, split="train", num_proc=8)
        prompt_ds = raw_ds.map(self.row_to_prompt,
                               fn_kwargs={'sys_prompt': sys_prompt},
                               remove_columns=raw_ds.column_names
                               )
        tok_ds    = prompt_ds.map(self.tokenize, batched=True)
        splits    = tok_ds.train_test_split(test_portion, seed=42)
        ds_train  = splits["train"]
        ds_test   = None
        if test_portion >= 0.0:
            ds_test = splits["test"]
        return ds_train, ds_test

    def train(self, train_ds: Dataset, test_ds: Dataset) -> None:
        """
        Creates the collator, arguments, and trainer to train the model.
        :param train_ds: Training dataset.
        :param test_ds: Testing dataset.
        :return: None
        """
        self.base_model.use_cache = False
        self.lora_model.train()

        data_collator = DataCollatorForCompletionOnlyLM(
            tokenizer            = self.tokenizer,
            instruction_template = f"{start_hd_tok}{usr_role}{end_hd_tok}",
            response_template    = f"{start_hd_tok}{bot_role}{end_hd_tok}")

        args = Seq2SeqTrainingArguments(
            output_dir                  = self.model_dir,
            per_device_train_batch_size = 2,
            per_device_eval_batch_size  = 1,
            gradient_accumulation_steps = 64,   # effective batch = 128
            num_train_epochs            = 10,
            learning_rate               = 2e-4,
            neftune_noise_alpha         = 0.1,
            logging_steps               = 2,
            bf16                        = True,
            bf16_full_eval              = True,
            save_strategy               = "epoch",
            eval_strategy               = "epoch",
            report_to                   = "none",
            label_names                 = ["labels"],
            metric_for_best_model       = "eval_loss",
            load_best_model_at_end      = True,
            eval_accumulation_steps     = 10,
            eval_on_start               = True,
            predict_with_generate       = True)

        early_stopping = EarlyStoppingCallback(
            early_stopping_patience  = 2,
            early_stopping_threshold = 0.001
        )

        trainer = Seq2SeqTrainer(
            model         = self.lora_model,
            args          = args,
            train_dataset = train_ds,
            eval_dataset  = test_ds,
            data_collator = data_collator,
            callbacks     = [early_stopping],
        )
        trainer.train()

    def inference(self, sys_prompt: str, contexts: list, query: str) -> str:
        """
        Takes a user prompt and builds the prompt to inference from the model.
        :param sys_prompt: System prompt.
        :param contexts: List of strings from the RAG database.
        :param query: User query.
        :return: The model output.
        """
        # Create prompt
        prompt = self.build_prompt(sys_prompt, contexts, query)

        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.lora_model.device)
        outputs = self.lora_model.generate(**inputs,
                                           max_new_tokens=256,
                                           do_sample=True,
                                           temperature=0.7,
                                           top_p=0.9,
                                           repetition_penalty=1.1,
                                           no_repeat_ngram_size=4,
                                           eos_token_id=self.tokenizer.eos_token_id)
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        return response