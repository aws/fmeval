## Foundation Model Evaluations Library
FMEval is a library to evaluate Large Language Models (LLMs) and select the best LLM
for your use case. The library can help evaluate LLMs for the following tasks:
* Open-ended generation - the production of natural language as a response to general prompts that do not have a
  pre-defined structure.
* Text summarization - summarizing the most important parts of a text, shortening a text while preserving its meaning.
* Question Answering - the generation of a relevant and accurate response to a question.
* Classification - assigning a category, such as a label or score, to text based on its content.

The library contains the following:
* Implementation of popular metrics (eval algorithms) such as Accuracy, Toxicity, Semantic Robustness and
  Prompt Stereotyping for evaluating LLMs across different tasks.
* Implementation of the ModelRunner interface. ModelRunner encapsulates the logic for invoking LLMs, exposing a predict
  method that greatly simplifies interactions with LLMs within eval algorithm code. The interface can be extended by
  the user for their LLMs.
  We have built-in support for AWS SageMaker Jumpstart Endpoints, AWS SageMaker Endpoints and Bedrock Models.

## Installation
To install the package from PIP you can simply do:

```
pip install fmeval
```

### Troubleshooting
If you you run into the error `error: can't find Rust compiler` while installing on a Mac, please try running the steps below.

```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup install 1.72.1
rustup default 1.72.1-aarch64-apple-darwin
rustup toolchain remove stable-aarch64-apple-darwin
rm -rf $HOME/.rustup/toolchains/stable-aarch64-apple-darwin
mv $HOME/.rustup/toolchains/1.72.1-aarch64-apple-darwin $HOME/.rustup/toolchains/stable-aarch64-apple-darwin
```


## Usage
You can see examples of running evaluations on your LLMs with built-in or custom datasets in
the [examples folder](https://github.com/aws/fmeval/tree/main/examples).

Main steps for using fmeval are:
1. Create a [ModelRunner](https://github.com/aws/fmeval/blob/main/src/fmeval/model_runners/model_runner.py)
   which can perform invocations on your LLM. We have built-in support for
   [AWS SageMaker Jumpstart Endpoints](https://github.com/aws/fmeval/blob/main/src/fmeval/model_runners/sm_jumpstart_model_runner.py),
   [AWS SageMaker Endpoints](https://github.com/aws/fmeval/blob/main/src/fmeval/model_runners/sm_model_runner.py)
   and [AWS Bedrock Models](https://github.com/aws/fmeval/blob/main/src/fmeval/model_runners/bedrock_model_runner.py).
   You can also extend the ModelRunner interface for any LLMs hosted anywhere.
2. Use any of the supported [eval_algorithms](https://github.com/aws/fmeval/tree/main/src/fmeval/eval_algorithms).
```
eval_algo = get_eval_algorithm("toxicity", ToxicityConfig())
eval_output = eval_algo.evaluate(model=model_runner)
```
*Note: You can update the default eval config parameters for your specific use case.*

### Using a custom dataset for an evaluation
We have our built-in datasets configured, which are consumed for computing the scores in eval algorithms.
You can choose to use a custom dataset in the following manner.
1. Create a [DataConfig](https://github.com/aws/fmeval/blob/main/src/fmeval/data_loaders/data_config.py)
   for your custom dataset
```
config = DataConfig(
    dataset_name="custom_dataset",
    dataset_uri="./custom_dataset.jsonl",
    dataset_mime_type="application/jsonlines",
    model_input_location="question",
    target_output_location="answer",
)
```

2. Use an eval algorithm with a custom dataset
```
eval_algo = get_eval_algorithm("toxicity", ToxicityConfig())
eval_output = eval_algo.evaluate(model=model_runner, dataset_config=config)
```

*Please refer to [code documentation](http://aws.github.io/fmeval) and
[examples]((https://github.com/aws/fmeval/tree/main/examples)) for understanding other details around the usage of
eval algorithms.*

## Development

### Setup
Once a virtual environment is set up with python3.10, run the following command to install all dependencies:
```
./devtool all
```

### Adding python dependencies
We use [poetry](https://python-poetry.org/docs/) to manage python dependencies in this project. If you want to add a new
dependency, please update the [pyproject.toml](./pyproject.toml) file, and run the `poetry update` command to update the
`poetry.lock` file (which is checked in).

Other than this step above to add dependencies, everything else should be managed with devtool commands.

### Adding your own Eval Algorithm

*Details TBA*

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
