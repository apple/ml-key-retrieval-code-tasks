# Synthetic data generator for multi-step key retrieval tasks

This folder contains code to generate synthetic data for one-step, two-step, three-step, and concatenation tasks.

These tasks are used in the following paper: <https://arxiv.org/abs/2407.21049>.

Install dependencies with:

```bash
pip install click datasets transformers
```

or if you want to use the dependency versions specified in requirements.txt do:

```bash
pip install -r requirements.txt
```

To generate the dataset for the main experiment from the paper, run:

```bash
python generate_data.py krc
```

To generate the dataset for the experiment on improving performance with call graph comments, run:

```bash
python generate_data.py krfix
python generate_data.py krfix_one_hop
```

The generated dataset will be saved to the `data` folder.

The dataset will contain a set of gzip-compressed JSON. Each JSON is an array of prompts and associated metadata. Below is an example:

```js
{
  // The prompt
  "prompt": "<the prompt string>",
  // A prefix regex to constraint decoding
  "force_decode_regex": "^[ \t]*(['\"]|$)",
  "metadata": {
    // The name of the model used to tokenize the prompt
    "model_name": "bigcode/starcoderbase",

    // The expected output string (as a python string literal)
    "expected": " \"eooyfwmxln\"",
    // Total number of tokens in the prompt, with the current model's tokenizer
    "prompt_token_count": 7929,
    // The permutation of the task-relevant snippets
    "permutation": [0, 1],
    // The positions of the task-relevant snippets among all snippets
    "positions": [2, 10],
    // The string ranges of each task-relevant snippet, always in the original order (before any permutation).
    // The snippet can be retrieved with prompt[range[0]:range[1]]
    "string_ranges": [
      [1295, 1339],
      [6032, 6079]
    ],
    // The token ranges of each task-relevant snippet, always in the original order (before any permutation)
    "token_ranges": [
      [351, 373],
      [1834, 1860]
    ],
    // The task
    "variant": "two-step",
    // The number of distractor functions
    "num_distractors": 1,
    // The max number of tokens in the prompt
    "max_prompt_tokens": 8000,

    // The max number of snippets from HumanEval (a very large number effectively removes the limit)
    "max_humaneval_snippets": 1000000,
    // The string length range of HumanEval snippets to include
    "humaneval_min_length": 250,
    "humaneval_max_length": 1000,

    // For krfix, the call graph comment type and template
    "call_graph_comment_type": "calls,called_by",
    "call_graph_template_variant": "calls_called_by"

    // Configuration of the task snippets (fixed for the experiment)
    "return_type": "string",
    "return_length": 10,
    "function_name": "random",
    "function_name_part_length": 6,
    "function_name_min_parts": 2,
    "function_name_max_parts": 3,
  }
}
```
