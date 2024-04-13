import argparse
import os
import time
from abc import ABC, abstractmethod
from collections import Counter
from dotenv import load_dotenv
from typing import List

import cohere
import pandas as pd
from openai import OpenAI
from sacrebleu.metrics import BLEU, CHRF
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

bleu = BLEU()
chrf = CHRF()
load_dotenv()


def bleu_key(s, r):
    return -chrf.corpus_score([s], [[r]]).score - bleu.corpus_score([s], [[r]]).score


def bleu_sort_inner(reference, series):
    return [bleu_key(sent, reference) for sent in series.to_list()]


def bleu_sort(df, ref, key="Source"):
    return df.sort_values(by=key, key=lambda s: bleu_sort_inner(ref[key], s))


class Model(ABC):
    @abstractmethod
    def generate_choices(self, prompt, n=1):
        pass


class OpenAIModel(Model):
    def __init__(self, model, api_key, temperature=0.1, max_tokens=50):
        self._model = model
        self._client = OpenAI(api_key=api_key)
        self._temperature = temperature
        self._max = max_tokens

    def generate_choices(self, prompt, n=1) -> List[str]:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self._temperature,
            max_tokens=self._max,
            n=n,
        )
        return [c.message.content for c in response.choices]


class CohereModel(Model):
    def __init__(self, model, api_key, temperature=0.3, max_tokens=50):
        self._model = model
        self._client = cohere.Client(api_key=api_key)
        self._temperature = temperature
        self._max = max_tokens

    def generate_choices(self, prompt, n=1) -> List[str]:
        response = self._client.chat(
            model=self._model,
            message=prompt,
            temperature=self._temperature,
            max_tokens=self._max,
        )
        return [response.text]


class HfModel(Model):
    def __init__(self, model):
        self._tokenizer = AutoTokenizer.from_pretrained(model)
        self._model = AutoModelForCausalLM.from_pretrained(model, device_map="auto")

    def generate_choices(self, prompt, n=1) -> List[str]:
        inputs = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], return_tensors="pt", return_dict=True
        ).to("cuda")
        # for _ in range(n):
        print(
            self._tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)
        )
        output = self._model.generate(**inputs, max_new_tokens=50)
        return self._tokenizer.batch_decode(output, skip_special_tokens=True)


def backoff(df, row):
    examples = []
    feats = "|".join(row["Change"].split("; "))
    for feat in feats.split("|"):
        if len(df[df.Change == feat]) > 0:
            examples.append((bleu_sort(df[df.Change == feat], row).head(3)))
        else:
            examples.append((bleu_sort(df[df.Change.str.contains(feat)], row).head(1)))
    more = df[df.Change.str.contains(feats)]
    examples.append(bleu_sort(more, row).head(8))
    examples = pd.concat(examples)
    return examples


def make_prompts(split, lang) -> tuple[List[tuple[str, str, str]], pd.DataFrame]:
    df: pd.DataFrame = pd.read_csv(f"data/{lang}-train.tsv", sep="\t")
    df["Change"] = df["Change"].str.replace(", ", "; ")
    df_test = pd.read_csv(f"data/{lang}-{split}.tsv", sep="\t")
    df_test["Change"] = df_test["Change"].str.replace(", ", "; ")

    columns = ["Source", "Change", "Target"]

    MIN_EXACT_MATCHES = 3
    advice = ""
    dataset = []
    for _, row in tqdm(df_test.iterrows()):
        if len(df[df.Change == row.Change]) < MIN_EXACT_MATCHES:
            examples = backoff(df, row)
        else:
            examples = bleu_sort(df[df.Change == row.Change], row).head(10)

        examples = examples[columns]
        prompt = f"{advice}Here's some examples.\n{examples.to_csv(index=False)}\nNow fill in the third column:\n{row['Source']},{row['Change']},"
        dataset.append((prompt, row.Source, row.Change))
    return dataset, df_test


def main(model, split, lang):
    dataset, df_test = make_prompts(split, lang)
    preds = []
    for i, (prompt, source, _) in tqdm(enumerate(dataset)):
        if i % 10 == 0 and i > 0:
            if isinstance(model, CohereModel):
                time.sleep(60)
            elif isinstance(model, OpenAIModel):
                time.sleep(2)
        choices = model.generate_choices(prompt, n=1)
        choices = [c.partition("\n")[0] for c in choices]
        choices = [c.split(",")[-1] if len(c.split(",")) > 1 else c for c in choices]
        tops = [c for c, _ in Counter(choices).most_common(1)]
        tops.sort(key=lambda x: bleu_key(x, source))
        preds.append(tops[0])
    if split == "dev":
        df_test["Predicted"] = preds
        df_test["prompt"] = [p for p, _, _ in dataset]
    elif split == "test":
        df_test["Target"] = preds
    df_test.Change = df_test.Change.str.replace("; ", ", ")
    print(df_test.to_csv(sep="\t", index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("lang", type=str)
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "test"])
    args = parser.parse_args()
    if args.model in ["gpt-3.5-turbo", "gpt-4"]:
        model = OpenAIModel(args.model, os.getenv("OPENAI_API_KEY"))
    elif args.model == "cohere":
        model = CohereModel("command-r-plus", os.getenv("COHERE_API_KEY"))
    else:
        model = HfModel(args.model)
    main(model, args.split, args.lang)
