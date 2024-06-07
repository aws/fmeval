import logging
from typing import Optional, List, Dict, Any

from fmeval.constants import (
    DatasetColumns,
    MEAN,
)
from fmeval.data_loaders.util import get_dataset
from fmeval.data_loaders.data_config import DataConfig
from fmeval.eval_algorithms.save_strategy import SaveStrategy
from fmeval.eval_algorithms.util import get_dataset_configs
from fmeval.eval_algorithms.common import evaluate_dataset, get_default_judge_model
from fmeval.eval_algorithms.eval_algorithm import EvalAlgorithmInterface, EvalAlgorithmConfig
from fmeval.eval_algorithms import (
    EvalAlgorithm,
    EvalOutput,
    EvalScore,
)
from fmeval.eval_algorithms.util import validate_dataset
from fmeval.model_runners.model_runner import ModelRunner
from fmeval.transforms.common import GetModelOutputs, GeneratePrompt
from fmeval.transforms.transform import Transform
from fmeval.transforms.transform_pipeline import TransformPipeline
from fmeval.transforms.util import validate_call
from fmeval.util import get_eval_results_path

logger = logging.getLogger(__name__)

FAITHFULNESS = EvalAlgorithm.FAITHFULNESS.value


LONG_FORM_ANSWER_PROMPT = """\
Human: Given a question and answer, create one or more statements from each sentence in the given answer. For each statement, please start with "Statement:".
question:$question
answer: $answer
statements:\n Assistant:"""  # noqa: E501


NLI_STATEMENTS_MESSAGE = """
Human: Prompt: Natural language inference
Consider the given context and following statements, then determine whether they are supported by the information present in the context.Provide a brief explanation for each statement before arriving at the verdict (Yes/No). Provide a final verdict for each statement in order at the end in the given format. Do not deviate from the specified format.

Context:\nJohn is a student at XYZ University. He is pursuing a degree in Computer Science. He is enrolled in several courses this semester, including Data Structures, Algorithms, and Database Management. John is a diligent student and spends a significant amount of time studying and completing assignments. He often stays late in the library to work on his projects.
statements:\n1. John is majoring in Biology.\n2. John is taking a course on Artificial Intelligence.\n3. John is a dedicated student.\n4. John has a part-time job.\n5. John is interested in computer programming.\n
Answer:
1. John is majoring in Biology.
Explanation: John's major is explicitly mentioned as Computer Science. There is no information suggesting he is majoring in Biology.  Verdict: No.
2. John is taking a course on Artificial Intelligence.
Explanation: The context mentions the courses John is currently enrolled in, and Artificial Intelligence is not mentioned. Therefore, it cannot be deduced that John is taking a course on AI. Verdict: No.
3. John is a dedicated student.
Explanation: The prompt states that he spends a significant amount of time studying and completing assignments. Additionally, it mentions that he often stays late in the library to work on his projects, which implies dedication. Verdict: Yes.
4. John has a part-time job.
Explanation: There is no information given in the context about John having a part-time job. Therefore, it cannot be deduced that John has a part-time job.  Verdict: No.
5. John is interested in computer programming.
Explanation: The context states that John is pursuing a degree in Computer Science, which implies an interest in computer programming. Verdict: Yes.
Final verdict for each statement in order: No. No. Yes. No. Yes.
context:\n$context
statements:\n$statements
Answer: Assistant:
"""  # noqa: E501


class FaithfulnessScore(Transform):
    """This transform augments its input record with the computed faithfulness score.

    See the docstring for `Faithfulness` for more details regarding the score itself.
    """

    def __init__(
        self,
        output_key: str = FAITHFULNESS,
    ):
        """FaithfulnessScore initializer.

        :param output_key: The key corresponding to the faithfulness score that
            will be added to the input record.
        """
        super().__init__(output_key)
        self.register_input_output_keys(
            input_keys=[],
            output_keys=[output_key],
        )
        self.output_key = output_key

    @validate_call
    def __call__(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Augment the input record with the computed faithfulness score.

        :param record: The input record.
        :returns: The input record, with the faithfulness score added in.
        """
        verdict_output = record["raw_verdicts"]
        statements = record["statements"]
        record[self.output_key] = self._get_score(verdict_output, statements)
        return record

    @staticmethod
    def _get_score(verdict_output: str, statements: str) -> int:
        """Given generated statements and if it can be inferred from the given contexts, computer Faithfulness score.

        :param verdict_output:
        :param statements:
        :returns: 0 to 1. See the docstring for `Faithfulness` for more details
            on what these numerical values represent.
        """
        output = verdict_output.lower().strip()
        num_statements = len(statements.split("\n"))

        # calculate score TODO edge case num_statements is 0; edge case if no verdicts found
        final_answer = "Final verdicts in order:"
        final_answer = final_answer.lower()
        if output.find(final_answer) != -1:
            output = output[output.find(final_answer) + len(final_answer) :]
            score = sum(0 if "yes" in answer else 1 for answer in output.strip().split(".") if answer != "")
            score = score / num_statements
        else:
            score = max(0, output.count("verdict: yes")) / num_statements

        return score


class GetStatements(Transform):
    """This transform gets statements string from raw statements."""

    def __init__(self, input_key: str, output_key: str, judge_model: ModelRunner):
        """Initializer.
        :param input_key: The key corresponding to prompt to get statements.
        :param output_key: The key corresponding to the statements, which gets added to the record.
        """
        super().__init__(input_key, output_key, judge_model)
        self.register_input_output_keys(
            input_keys=[input_key],
            output_keys=[output_key],
        )
        self.input_key = input_key
        self.output_key = output_key
        self.judge_model = judge_model

    @validate_call
    def __call__(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Augment the input record with the computed mean.
        :param record: The input record.
        :returns: The input record with the mean added in.
        """
        get_raw_statements = GetModelOutputs(
            input_to_output_keys={self.input_key: ["raw_statements"]},
            model_runner=self.judge_model,
        )
        record_with_raw_statements = get_raw_statements(record)
        raw_statements = record_with_raw_statements["raw_statements"].split("\n")
        list_statements = [statement for statement in raw_statements if statement.startswith("Statement:")]
        statements_str: str = "\n".join([f"{i + 1}.{st}" for i, st in enumerate(list_statements)])
        record[self.output_key] = statements_str
        return record


class Faithfulness(EvalAlgorithmInterface):
    """ """

    eval_name = EvalAlgorithm.FAITHFULNESS.value

    def __init__(self):
        """Faithfulness initializer."""
        super().__init__(EvalAlgorithmConfig())

    @staticmethod
    def _build_pipeline(
        judge_model: ModelRunner,
        long_form_prompt_template: str,
        nli_statements_prompt_template: str,
    ) -> TransformPipeline:
        gen_long_form_prompt = GeneratePrompt(
            input_keys=[],
            output_keys=["long_form_prompt"],
            prompt_template=long_form_prompt_template,
            placeholder_keys_dict={
                "question": DatasetColumns.MODEL_INPUT.value.name,
                "answer": DatasetColumns.MODEL_OUTPUT.value.name,
            },
        )
        get_statements = GetStatements(
            input_key="long_form_prompt",
            output_key="statements",
            judge_model=judge_model,
        )
        gen_nli_statements_prompt = GeneratePrompt(
            input_keys=[],
            output_keys=["nli_statements_prompt"],
            prompt_template=nli_statements_prompt_template,
            placeholder_keys_dict={"context": DatasetColumns.TARGET_CONTEXT.value.name, "statements": "statements"},
        )
        get_raw_verdicts = GetModelOutputs(
            input_to_output_keys={"nli_statements_prompt": ["raw_verdicts"]},
            model_runner=judge_model,
        )
        compute_score = FaithfulnessScore()
        transforms = [gen_long_form_prompt, get_statements, gen_nli_statements_prompt, get_raw_verdicts, compute_score]
        pipeline = TransformPipeline(transforms)
        return pipeline

    def evaluate_sample(
        self,
        model_input: str,
        model_output: str,
        target_context: str,
        judge_model: Optional[ModelRunner] = None,
        long_form_prompt_template: str = LONG_FORM_ANSWER_PROMPT,
        nli_statements_prompt_template: str = NLI_STATEMENTS_MESSAGE,
    ) -> List[EvalScore]:  # type: ignore[override]
        """Compute the faithfulness score on a single sample.

        :param model_input: The expected responses from the model.
        :param model_output: The output of the model being evaluated.
        :param target_context: The relevant context retrieved from RAG system.
        :param judge_model: An instance of ModelRunner representing the judge model to be used.
            If this argument is None, default judge model will be provided.
        :param long_form_prompt_template:
        :param nli_statements_prompt_template:
        :return: A single-element list containing an EvalScore corresponding to the faithfulness score.
        """
        if not judge_model:
            judge_model = get_default_judge_model()

        sample = {
            DatasetColumns.MODEL_INPUT.value.name: model_input,
            DatasetColumns.MODEL_OUTPUT.value.name: model_output,
            DatasetColumns.TARGET_CONTEXT.value.name: target_context,
        }
        pipeline = self._build_pipeline(
            judge_model=judge_model,
            long_form_prompt_template=long_form_prompt_template,
            nli_statements_prompt_template=nli_statements_prompt_template,
        )
        result = pipeline.execute_record(sample)
        return [EvalScore(name=FAITHFULNESS, value=result[FAITHFULNESS])]

    def evaluate(
        self,
        judge_model: Optional[ModelRunner] = None,
        dataset_config: Optional[DataConfig] = None,
        num_records: int = 300,
        save: bool = False,
        save_strategy: Optional[SaveStrategy] = None,
        long_form_prompt_template: str = LONG_FORM_ANSWER_PROMPT,
        nli_statements_prompt_template: str = NLI_STATEMENTS_MESSAGE,
    ) -> List[EvalOutput]:
        """Compute the faithfulness score on one or more datasets.

        :param judge_model: An instance of ModelRunner representing the judge model to be used.
            If this argument is None, default judge model will be provided.
        :param dataset_config: Configures the single dataset used for evaluation.
            If not provided, evaluations will be run on all of this algorithm's built-in datasets.
        :param num_records: The number of records to be sampled randomly from the input dataset(s)
            used to perform the evaluation(s). Note that the default value is 300, rather than
            100, as it is for the rest of the built-in algorithms. This is because there
            are 15 categories for faithfulness, and if only 100 samples are used, there
            will be categories with very few samples.
        :param save: If set to true, prompt responses and scores will be saved to a file.
        :param save_strategy: Specifies the strategy to use the save the localized outputs of the evaluations. If not
            specified, it will save it to the path that can be configured by the EVAL_RESULTS_PATH environment variable.
            If that environment variable is also not configured, it will be saved to the default path `/tmp/eval_results/`.
        :param long_form_prompt_template:
        :param nli_statements_prompt_template:

        :return: A list of EvalOutput objects.
        """
        dataset_configs = get_dataset_configs(dataset_config, self.eval_name)
        eval_outputs = []
        if not judge_model:
            judge_model = get_default_judge_model()
        for dataset_config in dataset_configs:
            dataset = get_dataset(dataset_config, num_records)
            validate_dataset(
                dataset,
                [
                    DatasetColumns.MODEL_INPUT.value.name,
                    DatasetColumns.MODEL_OUTPUT.value.name,
                    DatasetColumns.TARGET_CONTEXT.value.name,
                ],
            )
            eval_output = evaluate_dataset(
                dataset=dataset,
                pipeline=self._build_pipeline(judge_model, long_form_prompt_template, nli_statements_prompt_template),
                dataset_name=dataset_config.dataset_name,
                eval_name=self.eval_name,
                metric_names=[FAITHFULNESS],
                eval_results_path=get_eval_results_path(),
                agg_method=MEAN,
                save=save,
                save_strategy=save_strategy,
            )
            eval_outputs.append(eval_output)
        return eval_outputs
