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

```sh
pip install fmeval
```

## Usage
You can see examples of running evaluations on your LLMs with built-in or custom datasets in
the [examples folder](https://github.com/aws/fmeval/tree/main/examples).

The main steps for using `fmeval` are:
1. Create a `ModelRunner` which can perform invocation on your LLM. `fmeval` provides built-in support for Amazon SageMaker Endpoints and JumpStart LLMs. You can also extend the `ModelRunner` interface for any LLMs hosted anywhere.
2. Use any of the supported [eval_algorithms](https://github.com/aws/fmeval/tree/main/src/fmeval/eval_algorithms).

For example,
```python
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
```python
config = DataConfig(
    dataset_name="custom_dataset",
    dataset_uri="./custom_dataset.jsonl",
    dataset_mime_type="application/jsonlines",
    model_input_location="question",
    target_output_location="answer",
)
```

2. Use an eval algorithm with a custom dataset
```python
eval_algo = Toxicity(ToxicityConfig())
eval_output = eval_algo.evaluate(model=model_runner, dataset_config=config)
```

*Please refer to the [developer guide](https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-foundation-model-evaluate-auto.html) and
[examples](https://github.com/aws/fmeval/tree/main/examples) for more details around the usage of
eval algorithms.*

## Telemetry
`fmeval` has telemetry enabled for tracking the usage of AWS-provided/hosted LLMs.
This data is tracked using the number of SageMaker or JumpStart `ModelRunner` objects that get created.
Telemetry can be disabled by setting the `DISABLE_FMEVAL_TELEMETRY` environment variable to `true`.


## Troubleshooting

1. Users running `fmeval` on a Windows machine may encounter the error `OSError: [Errno 0] AssignProcessToJobObject() failed` when `fmeval` internally calls `ray.init()`. This OS error is a known Ray issue, and is detailed [here](https://github.com/ray-project/ray/issues/21994). Multiple users have reported that installing Python from the [official Python website](https://www.python.org/downloads/windows/) rather than the Microsoft store fixes this issue. You can view more details on limitations of running Ray on Windows on [Ray's webpage](https://docs.ray.io/en/latest/ray-overview/installation.html#windows-support).

2. If you run into the error `error: can't find Rust compiler` while installing `fmeval` on a Mac, please try running the steps below.

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup install 1.72.1
rustup default 1.72.1-aarch64-apple-darwin
rustup toolchain remove stable-aarch64-apple-darwin
rm -rf $HOME/.rustup/toolchains/stable-aarch64-apple-darwin
mv $HOME/.rustup/toolchains/1.72.1-aarch64-apple-darwin $HOME/.rustup/toolchains/stable-aarch64-apple-darwin
```

3. If you run into the error `ERROR: Cannot install fmeval==0.2.0, fmeval==0.2.1, fmeval==0.3.0, fmeval==0.4.0, fmeval==1.0.0, fmeval==1.0.1, fmeval==1.0.2, fmeval==1.0.3 and fmeval==1.1.0 because these package versions have conflicting dependencies` while installing `fmeval`, please try deactivating and recreating your virtual environment using the steps below. Make sure to replace `<your_virtual_env>` with the name of your actual virtual environment:

```sh
virtualenv --clear <your_virtual_env>
mkvirtualenv <your_virtual_env> -p python3.10
```

4. If you run into out of memory (OOM) errors, especially while running evaluations that use LLMs as evaluators like toxicity and
summarization accuracy, it is likely that your machine does not have enough memory to load the evaluator
models. By default, `fmeval` loads multiple copies of the model into memory to maximize parallelization, where the exact number depends on the number of cores on the machine. To reduce the number of models that get loaded in parallel, you can
set the environment variable `PARALLELIZATION_FACTOR` to a value that suits your machine.

## Development

### Setup and the use of `devtool`
Once you have created a virtual environment with python3.10, run the following command to set up the development environment:
```sh
./devtool install_deps_dev
./devtool install_deps
./devtool all
```

**Note**: If you are on a Mac, the `install_poetry_version` devtool command may fail when running the poetry installation script. If there is a failure, you should get error logs sent to a file with a name like `poetry-installer-error-cvulo5s0.log`. Open the logs, and if the error message looks like the following:
```
dyld[10908]: Library not loaded: @loader_path/../../../../Python.framework/Versions/3.10/Python
  Referenced from: <8A5DEEDB-CE8E-325F-88B0-B0397BD5A5DE> /Users/daniezh/Library/Application Support/pypoetry/venv/bin/python3
  Reason: tried: '/Users/daniezh/Library/Application Support/pypoetry/venv/bin/../../../../Python.framework/Versions/3.10/Python' (no such file), '/Library/Frameworks/Python.framework/Versions/3.10/Python' (no such file), '/System/Library/Frameworks/Python.framework/Versions/3.10/Python' (no such file, not in dyld cache)

Traceback:

  File "<string>", line 923, in main
  File "<string>", line 562, in run
```
then you will need to tweak the poetry installation script and re-run it.

Steps:
1. `curl -sSL https://install.python-poetry.org > poetry_script.py`
2. Change the `symlinks` argument in `builder = venv.EnvBuilder(clear=True, with_pip=True, symlinks=False)` to `True`. See mionker's comment [here](https://github.com/python-poetry/install.python-poetry.org/issues/56) for an explanation.
3. `python poetry_script.py --version 1.8.2` (where `1.8.2` is the version listed in `devtool`; this may change after the time of this writing).
4. Confirm installation via `poetry --version`

Additionally, if you already have an existing version of Poetry installed and want to install a new version, before you re-run the above command, you will need to uninstall Poetry:

`curl -sSL https://install.python-poetry.org | python3 - --uninstall`

Before submitting a PR, rerun `./devtool all` for testing and linting. It should run without errors.

### Adding python dependencies
We use [poetry](https://python-poetry.org/docs/) to manage python dependencies in this project. If you want to add a new
dependency, please update the [pyproject.toml](./pyproject.toml) file, and run the `poetry update` command to update the
`poetry.lock` file (which is checked in).

Other than this step to add dependencies, use devtool commands for installing dependencies, linting and testing. Execute the command `./devtool` without any arguments to see a list of available options.

### Adding your own evaluation algorithm and/or metrics

The evaluation algorithms and metrics provided by `fmeval` are implemented using `Transform` and `TransformPipeline` objects. You can leverage these existing tools to similarly implement your own metrics and algorithms in a modular manner.

Here, we provide a high-level overview of what these classes represent and how they are used. Specific implementation details can be found in their respective docstrings (see `src/fmeval/transforms/transform.py` and `src/fmeval/transforms/transform_pipeline.py`).

#### Preface
At a high level, an evaluation algorithm takes an initial tabular dataset consisting of a number of "records" (i.e. rows) and repeatedly transforms this dataset until the dataset either contains all the evaluation metrics, or at least all the intermediate data needed to compute said metrics. The transformations that get applied to the dataset inherently operate at a per-record level, and simply get applied to every record in the dataset to transform the dataset in full.

#### The `Transform` class
We represent the concept of a record-level transformation using the `Transform` class. `Transform` is a callable class where its `__call__` method takes a single argument, `record`, which represents the record to be transformed. A record is represented by a Python dictionary. To implement your own record-level transformation logic, create a concrete subclass of `Transform` and implement its `__call__` method.

**Example:**

Let's implement a `Transform` for a simple, toy metric.

```python
class NumSpaces(Transform):
    """
    Augments the input record (which contains some text data)
    with the number of spaces found in the text.
    """
    def __call__(self, record: Dict[str, Any]) -> Dict[str, Any]:
        input_text = record["input_text"]
        record["num_spaces"] = input_text.count(" ")
        return record
```

One issue with this simple example is that the keys used for the input text data and the output data are both hard-coded. This generally isn't desirable, so let's improve on our running example.

```python
class NumSpaces(Transform):
    """
    Augments the input record (which contains some text data)
    with the number of spaces found in the text.
    """

    def __init__(self, text_key, output_key):
        super().__init__(text_key, output_key)  # always need to pass all init args to superclass init
        self.text_key = text_key  # the dict key corresponding to the input text data
        self.output_key = output_key  # the dict key corresponding to the output data (i.e. number of spaces)

    def __call__(self, record: Dict[str, Any]) -> Dict[str, Any]:
        input_text = record[self.text_key]
        record[self.output_key] = input_text.count(" ")
        return record
```

Since `__call__` only takes a single argument, `record`, we pass the information regarding which keys to use for input and output data to `__init__` and save them as instance attributes. Note that all subclasses of `Transform` need to call `super().__init__` with all of their `__init__` arguments, due to low-level implementation details regarding how we apply the `Transform`s to the dataset.

#### The `TransformPipeline` class
While `Transform` encapsulates the logic for the record-level transformation, we still don't have a mechanism for applying the transform to a dataset. This is where `TransformPipeline` comes in. A `TransformPipeline` represents a sequence, or "pipeline", of `Transform` objects that you wish to apply to a dataset. After initializing a `TransformPipeline` with a list of `Transform`s, simply call its `execute` method on an input dataset.

**Example:**
Here, we implement a pipeline for a very simple evaluation. The steps are:
1. Construct LLM prompts from raw text inputs
2. Feed the prompts to a `ModelRunner` to get the model outputs
3. Compute the "number of spaces" metric we defined above

```python
# Use the built-in utility Transform for generating prompts
gen_prompt = GeneratePrompt(
    input_keys="model_input",
    output_keys="prompt",
    prompt_template="Answer the following question: $model_input",
)

# Use the built-in utility Transform for getting model outputs
model = ... # some ModelRunner
get_model_outputs = GetModelOutputs(
    input_to_output_keys={"prompt": ["model_output"]},
    model_runner=model,
)

# Our new metric!
compute_num_spaces = NumSpaces(
    text_key="model_output",
    output_key="num_spaces",
)

my_pipeline = TransformPipeline([gen_prompt, get_model_outputs, compute_num_spaces])
dataset = # load some dataset
dataset = my_pipeline.execute(dataset)
```

#### Conclusion
To implement new metrics, create a new `Transform` that encapsulates the logic for computing said metric. Since the logic for all evaluation algorithms can be represented as a sequence of different `Transform`s, implementing a new evaluation algorithm essentially amounts to defining a `TransformPipeline`. Please see the built-in evaluation algorithms for examples.
## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
