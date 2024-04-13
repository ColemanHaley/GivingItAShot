from itertools import product
LANGUAGES=["guarani", "maya", "bribri"]
SPLITS=["test"]
MODELS=["gpt-4", "gpt-3.5-turbo", "cohere"]
rule all:
  input:
    [f"output/results_{lang}_{split}_{model}.tsv" for lang, split, model in product(LANGUAGES, SPLITS, MODELS)]

rule results:
  output:
    "output/results_{lang}_{split}_{model}.tsv"

  shell:
    "python main.py --split {wildcards.split} --model {wildcards.model} {wildcards.lang} > {output}"
