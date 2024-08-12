#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

"""Generate data."""

import hashlib
import itertools
import json
import os
import subprocess
import traceback
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

import click


def generate_krc(max_prompt_tokens: int):
    models = [
        "mistralai/Mistral-7B-v0.1",
        "bigcode/starcoderbase",
        "bigcode/starcoderbase-1b",
        "bigcode/starcoderbase-7b",
        "bigcode/starcoder2-7b",
    ]

    variants = ["one-step", "two-step", "three-step", "concatenation"]

    distractors = [0, 1, 5]

    seeds = [100]

    num_key_functions_dict = {
        "one-step": 100,
        "two-step": 20,
        "three-step": 20,
        "concatenation": 20,
    }

    # Set max_position_combinations for three-step and concatenation so
    # the total number of prompts won't be too large.
    # For one-step, we don't really need to limit this.
    max_pos_combs_dict = {
        "one-step": None,
        "two-step": 150,  # max number = 20 * 2 * 150 = 6000
        "three-step": 50,  # max number = 20 * 6 * 50 = 6000
        "concatenation": 50,  # max number = 20 * 6 * 50 = 6000
    }

    combinations = itertools.product(models, variants, distractors, seeds)
    for model, variant, distractors, seed in combinations:
        mname = model.split("/")[-1]
        yield "krc", {
            "model_name": model,
            "variant": variant,
            "return_type": "string",
            "return_length": 10,
            "function_name": "random",
            "num_key_functions": num_key_functions_dict[variant],
            "max_position_combinations": max_pos_combs_dict.get(variant),
            "humaneval_max_length": 1000 if max_prompt_tokens >= 4000 else 500,
            "max_prompt_tokens": max_prompt_tokens,
            "max_humaneval_snippets": 1000000,
            "num_distractors": distractors,
            "seed": seed,
            "output": f"./data/krc/krc_{max_prompt_tokens}_{variant}_{mname}_{distractors}_distractors_seed{seed}.json.gz",
        }


def generate_krfix(max_prompt_tokens: int):
    models = [
        "mistralai/Mistral-7B-v0.1",
        "bigcode/starcoderbase",
        "bigcode/starcoderbase-1b",
        "bigcode/starcoderbase-7b",
        "bigcode/starcoder2-7b",
    ]

    variants = ["three-step", "concatenation"]

    distractors = [5]

    call_graph_comment_types = ["calls", "called_by", "calls,called_by"]
    call_graph_template_variants = ["calls_called_by", "function_names_only"]

    seeds = [100, 348]

    num_key_functions_dict = {
        "three-step": 5,
        "concatenation": 5,
    }

    max_pos_combs_dict = {
        "three-step": 15,
        "concatenation": 15,
    }

    combinations = itertools.product(
        models,
        variants,
        distractors,
        call_graph_comment_types,
        call_graph_template_variants,
        seeds,
    )
    for (
        model,
        variant,
        distractors,
        call_graph_comment_type,
        call_graph_template_variant,
        seed,
    ) in combinations:
        mname = model.split("/")[-1]
        yield "krfix", {
            "model_name": model,
            "variant": variant,
            "return_type": "string",
            "return_length": 10,
            "function_name": "random",
            "num_key_functions": num_key_functions_dict[variant],
            "max_position_combinations": max_pos_combs_dict.get(variant),
            "humaneval_max_length": 1000 if max_prompt_tokens >= 4000 else 500,
            "max_prompt_tokens": max_prompt_tokens,
            "max_humaneval_snippets": 1000000,
            "num_distractors": distractors,
            "call_graph_comment_type": call_graph_comment_type,
            "call_graph_template_variant": call_graph_template_variant,
            "seed": seed,
            "output": f"./data/krfix/krfix_{max_prompt_tokens}_{variant}_{mname}_{distractors}_distractors_{call_graph_comment_type}_{call_graph_template_variant}_seed{seed}.json.gz",
        }


def generate_krfix_one_hop(max_prompt_tokens: int):
    models = [
        "mistralai/Mistral-7B-v0.1",
        "bigcode/starcoderbase",
        "bigcode/starcoderbase-1b",
        "bigcode/starcoderbase-7b",
        "bigcode/starcoder2-7b",
    ]

    variants = ["three-step"]

    distractors = [5]

    call_graph_comment_types = [
        "calls_one_hop",
        "called_by_one_hop",
        "calls_one_hop,called_by_one_hop",
    ]
    call_graph_template_variants = ["calls_called_by", "function_names_only"]

    seeds = [100, 348]

    num_key_functions_dict = {
        "three-step": 5,
    }

    max_pos_combs_dict = {
        "three-step": 15,
    }

    combinations = itertools.product(
        models,
        variants,
        distractors,
        call_graph_comment_types,
        call_graph_template_variants,
        seeds,
    )
    for (
        model,
        variant,
        distractors,
        call_graph_comment_type,
        call_graph_template_variant,
        seed,
    ) in combinations:
        mname = model.split("/")[-1]
        yield "krfix_one_hop", {
            "model_name": model,
            "variant": variant,
            "return_type": "string",
            "return_length": 10,
            "function_name": "random",
            "num_key_functions": num_key_functions_dict[variant],
            "max_position_combinations": max_pos_combs_dict.get(variant),
            "humaneval_max_length": 1000 if max_prompt_tokens >= 4000 else 500,
            "max_prompt_tokens": max_prompt_tokens,
            "max_humaneval_snippets": 1000000,
            "num_distractors": distractors,
            "call_graph_comment_type": call_graph_comment_type,
            "call_graph_template_variant": call_graph_template_variant,
            "call_graph_comment_position": "before",
            "seed": seed,
            "output": f"./data/krfix_one_hop/krfix_one_hop_{max_prompt_tokens}_{variant}_{mname}_{distractors}_distractors_{call_graph_comment_type}_{call_graph_template_variant}_seed{seed}.json.gz",
        }


def tasks(configs) -> list[list[str]]:
    result = []
    for experiment, config in configs:
        args = [experiment]
        for key, value in config.items():
            if value is not None:
                args += [f"--{key}", str(value)]
        os.makedirs(os.path.dirname(config["output"]), exist_ok=True)
        result.append(args)
    return result


def run_tasks_in_parallel(task_list: list[list[str]]):
    def work(arguments: list[str]):
        os.makedirs("./logs", exist_ok=True)
        key = hashlib.sha1(json.dumps(arguments).encode("utf-8")).hexdigest()
        f_stdout = f"./logs/{key}.stdout"
        f_stderr = f"./logs/{key}.stderr"
        with open(f_stdout, "wb") as out, open(f_stderr, "wb") as err:
            out.write((json.dumps(arguments) + "\n\n").encode("utf-8"))
            err.write((json.dumps(arguments) + "\n\n").encode("utf-8"))
            success = False
            try:
                subprocess.check_call(
                    ["python", "data_generator.py"] + arguments,
                    stdout=out,
                    stderr=err,
                )
                success = True
            except:
                click.echo(
                    click.style("[Error]", fg="red")
                    + f" {' '.join(arguments)}, see {f_stderr}"
                )
                traceback.print_exc()
            if success:
                click.echo(
                    click.style("[Success]", fg="green") + f" {' '.join(arguments)}"
                )

    tp = ThreadPool(min(cpu_count(), 16))
    for arguments in task_list:
        tp.apply_async(work, (arguments,))

    tp.close()
    tp.join()


@click.group()
def cli():
    pass


@cli.command()
def krc():
    all_tasks = []
    all_tasks += tasks(generate_krc(2000))
    all_tasks += tasks(generate_krc(4000))
    all_tasks += tasks(generate_krc(8000))
    run_tasks_in_parallel(all_tasks)


@cli.command()
def krfix():
    all_tasks = []
    all_tasks += tasks(generate_krfix(2000))
    all_tasks += tasks(generate_krfix(4000))
    all_tasks += tasks(generate_krfix(8000))
    run_tasks_in_parallel(all_tasks)


@cli.command()
def krfix_one_hop():
    all_tasks = []
    all_tasks += tasks(generate_krfix_one_hop(2000))
    all_tasks += tasks(generate_krfix_one_hop(4000))
    all_tasks += tasks(generate_krfix_one_hop(8000))
    run_tasks_in_parallel(all_tasks)


if __name__ == "__main__":
    cli()
