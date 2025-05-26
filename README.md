# PDF Fine-tuned LLM
## Repository Structure
The repository is split into three main parts:
1) notebooks: The jupyter notebooks I used to develop the pipeline. These contain
detailed reasoning and explanation for different decisions made throughout the
process.
2) pipeline: The finalized pipeline module. This is designed to be idem-potent
and robust (though lacks logging) allowing someone to create a mostly "batteries-included"
application.
3) pdfs: Where the original PDF file(s) live.

## Getting Started
#### Install Requirements
I have only tested this on my computer with a 30-series nVidia GPU. If running on CPU,
you will need to change the line "torch~=2.7.0+cu118" in requirements.txt and remove
the "+cu118" part.
```shell
pip install -r requirements.txt
```

#### Create .env file(s)
Secrets are stored in .env files, which do not get uploaded to the repository.
You can use the included sample environment files to get started. You will need
to provide your own API keys.
1) Copy appropriate file into notebooks/pipeline.
2) Fill in the OpenAI and HuggingFace Hub API keys.

The pipeline is also configured using this file to provide a single place where 
tweaks and updates are made.

#### Notebooks
From this point, open each notebook sequentially and run through all the cells.

The notebook outputs have been included so you can just look at my outputs/notes
if you don't want to run the whole thing. Start to finish takes about ~30 minutes
on an RTX 3060 12-GB card.

#### Pipeline
From the "pipeline" directory, tun the following command to run the pipeline:
```shell
python main.py
```
From there you can watch the steps complete as it is printed to the console.
Once the model is trained, you will be able to interact with it by typing
into the console. This has been sped up a bit compared to the notebooks (less
evaluation metrics and multithreaded OpenAI calls) so it takes about 20 minutes
start-to-interaction on my 3060.
