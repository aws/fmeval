## Foundation Model Evaluations Library
`fmeval` is a library to evaluate Large Language Models (LLMs) in order to help select the best LLM
for your use case. The library evaluates LLMs for the following tasks:
* Open-ended generation - The production of natural human responses to text that does not have a pre-defined structure.
* Text summarization - The generation of a condensed summary retaining the key information contained in a longer text.
* Question Answering - The generation of a relevant and accurate response to an answer.
* Classification - Assigning a category, such as a label or score to text, based on its content.

The library contains
* Algorithms to evaluate LLMs for Accuracy, Toxicity, Semantic Robustness and
  Prompt Stereotyping across different tasks.
* Implementations of the `ModelRunner` interface. `ModelRunner` encapsulates the logic for invoking different types of LLMs, exposing a `predict` method to simplify interactions with LLMs within the eval algorithm code. We have built-in support for Amazon SageMaker Endpoints and JumpStart models. The user can extend the interface for their own model classes by implementing the `predict` method.

## Installation
`fmeval` is developed under python3.10. To install the package, simply run:

```
pip install fmeval
```

## Usage
You can see examples of running evaluations on your LLMs with built-in or custom datasets in
the [examples folder](https://github.com/aws/fmeval/tree/main/examples).

The main steps for using `fmeval` are:
1. Create a `ModelRunner` which can perform invocation on your LLM. `fmeval` provides built-in support for Amazon SageMaker Endpoints and JumpStart LLMs. You can also extend the `ModelRunner` interface for any LLMs hosted anywhere.
2. Use any of the supported [eval_algorithms](https://github.com/aws/fmeval/tree/main/src/fmeval/eval_algorithms).

For example,
```
from fmeval.eval_algorithms.toxicity import Toxicity, ToxicityConfig

eval_algo = Toxicity(ToxicityConfig())
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
eval_algo = Toxicity(ToxicityConfig())
eval_output = eval_algo.evaluate(model=model_runner, dataset_config=config)
```

*Please refer to the [developer guide](https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-foundation-model-evaluate-auto.html) and
[examples](https://github.com/aws/fmeval/tree/main/examples) for more details around the usage of
eval algorithms.*

## Troubleshooting

1. Users running `fmeval` on a Windows machine may encounter the error `OSError: [Errno 0] AssignProcessToJobObject() failed` when `fmeval` internally calls `ray.init()`. This OS error is a known Ray issue, and is detailed [here](https://github.com/ray-project/ray/issues/21994). Multiple users have reported that installing Python from the [official Python website](https://www.python.org/downloads/windows/) rather than the Microsoft store fixes this issue. You can view more details on limitations of running Ray on Windows on [Ray's webpage](https://docs.ray.io/en/latest/ray-overview/installation.html#windows-support).

2. If you run into the error `error: can't find Rust compiler` while installing `fmeval` on a Mac, please try running the steps below.

```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup install 1.72.1
rustup default 1.72.1-aarch64-apple-darwin
rustup toolchain remove stable-aarch64-apple-darwin
rm -rf $HOME/.rustup/toolchains/stable-aarch64-apple-darwin
mv $HOME/.rustup/toolchains/1.72.1-aarch64-apple-darwin $HOME/.rustup/toolchains/stable-aarch64-apple-darwin
```

3. If you run into out of memory (OOM) errors, especially while running evaluations that use LLMs as evaluators like toxicity and
summarization accuracy, it is likely that your machine does not have enough memory to load the evaluator
models. By default, `femval` loads multiple copies of the model into memory to maximize parallelization, where the exact number depends on the number of cores on the machine. To reduce the number of models that get loaded in parallel, you can
set the environment variable `PARALLELIZATION_FACTOR` to a value that suits your machine.

## Development

### Setup and the use of `devtool`
Once you have created a virtual environment with python3.10, run the following command to set up the development environment:
```
./devtool install_deps_dev
./devtool install_deps
./devtool all
```

Before submitting a PR, rerun `./devtool all` for testing and linting. It should run without errors.

### Adding python dependencies
We use [poetry](https://python-poetry.org/docs/) to manage python dependencies in this project. If you want to add a new
dependency, please update the [pyproject.toml](./pyproject.toml) file, and run the `poetry update` command to update the
`poetry.lock` file (which is checked in).

Other than this step to add dependencies, use devtool commands for installing dependencies, linting and testing. Execute the command `./devtool` without any arguments to see a list of available options.

### Adding your own Eval Algorithm

*Details TBA*

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
