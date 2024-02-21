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
* Implementations of the `ModelRunner` interface. `ModelRunner` encapsulates the logic for invoking different types of LLMs, exposing a `predict`
  method to simplify interactions with LLMs within the eval algorithm code. The interface can be extended by
  the user for their own model classes.
  We have built-in support for AWS SageMaker Jumpstart Endpoints, AWS SageMaker Endpoints and Bedrock Models.

## Installation
`fmeval` is developed under python3.10. To install the package from PIP you can simply do:

```
pip install fmeval
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
[examples]((https://github.com/aws/fmeval/tree/main/examples)) for more details around the usage of
eval algorithms.*

## Development

### Setup
Once a virtual environment is set up with python3.10, run the following command to install all dependencies:
```
./devtool all
```
If this fails, you might need to install some dependencies first:
```
./devtool install_deps_dev
./devtool install_deps
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
