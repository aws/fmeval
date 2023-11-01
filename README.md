## AWS Foundation Model Evaluations Library

SageMaker Clarify is offering a new feature, Foundation Model Evaluation (FME), in a beta release to help you evaluate 
and select the best large language models (LLMs) for your use case. A foundation model serves as a starting point, from 
which you can develop downstream natural language processing (NLP) applications including conversational artificial 
intelligence (AI), content generation and text summarization. FME can help you develop and select the best LLM for your 
use case by evaluating your model using metrics that are compared to standard industry benchmarks. FME can also help 
safeguard against providing toxic, harmful or otherwise poor responses to your customers. Lastly, FME can help you 
comply with international guidelines around responsible generative AI including ISO 42001. FME can help you evaluate 
LLMs for the following tasks:

* **Open-ended generation** - the production of natural human responses to general questions that do not have a pre-defined structure. 
* **Text summarization** - the verbatim extraction of a few pieces of highly relevant text (extraction) or the condensed summarization of the original text (abstraction).
* **Question Answering** - the generation of a relevant and accurate response to a question.
* **Classification** - assigning a category, such as a label or score, to text based on its content.

The following sections show how to configure an evaluation.

## Configure a foundation model evaluation

You can configure your foundation model evaluation and customize it for your use case. Your configuration will depend 
both on the kind of task that your foundation model is built to predict, and how you want to evaluate it. The following
steps show you how to set up your environment and run a factual knowledge evaluation algorithm with a SageMaker 
JumpStart endpoint. Then, we show configurations and parameters to run evaluations for open-ended generation, text 
summarization, question answering, and task classification tasks.

### Step 1: Setup your environment
To get started, install the amazon_fmeval package in your development environment, as shown in the following code example.

```
pip install amazon-fmeval
```

### Step 2: Create the ModelRunner
The `amazon_fmeval` library can evaluate any LLM. The only requirement for the model is to be wrapped in the ModelRunner
interface. The ModelRunner interface has one abstract method: `def predict(self, prompt: str) -> Tuple[Optional[str], Optional[float]]:`.
This method takes in the prompt that has to be fed to the model, and returns a tuple of (output text, input log 
probability). 

The `amazon-fmeval` package has built-in implementations of ModelRunner interface:
1. [`SageMakerModelRunner`](https://fantastic-waddle-n8nvqmv.pages.github.io/src/amazon_fmeval/model_runners/sm_model_runner.html), 
2. SageMaker JumpStart models with [`JumpStartModelRunner`](https://fantastic-waddle-n8nvqmv.pages.github.io/src/amazon_fmeval/model_runners/sm_jumpstart_model_runner.html).
3. AWS Bedrock models with [Bedrock ModelRunner](https://fantastic-waddle-n8nvqmv.pages.github.io/src/amazon_fmeval/model_runners/bedrock_model_runner.html)
The different built-in ModelRunners can be found [here]().  

### Step 3: Evaluate your model



## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
