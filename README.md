This repository contains the code for my submission to the AmericasNLP2024 Shared Task on creating educational materials for indigenous languages.

To run the code, you need to set the environment variables OPENAI_API_KEY and COHERE_API_KEY (it's designed to handle a free Cohere key, but you may need more than 1 account to run all experiments).

Then, simply run `snakemake -j 1` to replicate the results (more jobs can result in rate limiting issues).
