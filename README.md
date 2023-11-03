## Foundation Model Evaluations Library
FMEval is a library to evaluate Large Language Models (LLMs), to help evaluate and select the best large language models (LLMs)
for your use case.  The library can help evaluate LLMs for the following tasks:
* Open-ended generation - the production of natural human responses to general questions that do not have a
  pre-defined structure.
* Text summarization - the verbatim extraction of a few pieces of highly relevant text (extraction) or the condensed
  summarization of the original text (abstraction).
* Question Answering - the generation of a relevant and accurate response to a question.
* Classification - assigning a category, such as a label or score, to text based on its content.

The library contains the following:
* Implementation of popular metrics (eval algorithms) such as Accuracy, Toxicity, Semantic Robustness and
  Prompt Stereotyping or evaluating LLMs across different tasks.
* Implementation of ModelRunner interface. ModelRunner encapsulates the logic for invoking LLMs, exposing a predict
  method that greatly simplifies interactions with LLMs within eval algorithm code. The interface can be extended by
  the user for their LLMs.
  We have built-in support for AWS SageMaker Jumpstart Endpoints, AWS SageMaker Endpoints and Bedrock Models.

## Installation
To install the package from PIP you can simply do:

```
pip install fmeval
```
*Note: A PyPi package is currently not available. Please reach out to the team for beta preview whl distribution*

## Usage
You can see examples of running evaluations on your LLMs with built-in or custom datasets in
the [examples folder](https://github.com/aws/amazon-fmeval/tree/main/examples).

Main steps for using fmeval are:
1. Create a [ModelRunner](https://github.com/aws/amazon-fmeval/blob/main/src/amazon_fmeval/model_runners/model_runner.py)
   which can can perform invocations on your LLM. We have built-in support for
   [AWS SageMaker Jumpstart Endpoints](https://github.com/aws/amazon-fmeval/blob/main/src/amazon_fmeval/model_runners/sm_jumpstart_model_runner.py),
   [AWS SageMaker Endpoints](https://github.com/aws/amazon-fmeval/blob/main/src/amazon_fmeval/model_runners/sm_model_runner.py)
   and [AWS Bedrock Models](https://github.com/aws/amazon-fmeval/blob/main/src/amazon_fmeval/model_runners/bedrock_model_runner.py).
   You can also extend the ModelRunner interface for any LLMs hosted anywhere..
2. Use any of the supported [eval_algorithms](https://github.com/aws/amazon-fmeval/tree/main/src/amazon_fmeval/eval_algorithms).
```
eval_algo = get_eval_algorithm("toxicity", ToxicityConfig())
eval_output = eval_algo.evaluate(model=model_runner)
```
*Note: You can update the default eval config parameters for your specific use case.*

### Using a custom dataset for an evaluation
We have our built-in datasets configured, which are consumed for computing the scores in eval algorithms.
You can choose to use a custom dataset in the following manner.
1. Create a [DataConfig](https://github.com/aws/amazon-fmeval/blob/main/src/amazon_fmeval/data_loaders/data_config.py)
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

2. Use eval algorithm with custom dataset
```
eval_algo = get_eval_algorithm("toxicity", ToxicityConfig())
eval_output = eval_algo.evaluate(model=model_runner, dataset_config=config)
```

*Please refer to [code documentation](https://fantastic-waddle-n8nvqmv.pages.github.io/src/amazon_fmeval.html) and
[examples]((https://github.com/aws/amazon-fmeval/tree/main/examples)) for understanding other details around usage of
eval algorithms*

## Development

### Setup
Once a virtual environment is set up with python3.10, run the following command to install all dependencies:
```
./devtool all
```

### Adding python dependencies
We use [poetry](https://python-poetry.org/docs/) to manage python dependencies in this project. If you want to add a new
dependency, please update the [pyproject.toml](./pyproject.toml) file, and run `poetry update` command to update the
`poetry.lock` file (which is checked-in).

Other than this step above to add dependencies, everything else should be managed with devtool commands.

### Adding your own Eval Algorithm

*Details TBA*

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
