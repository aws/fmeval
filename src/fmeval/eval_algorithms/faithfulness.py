import logging
from typing import Optional, List, Dict, Any, Tuple

from fmeval.constants import (
    DatasetColumns,
    MEAN,
)
from fmeval.data_loaders.util import get_dataset
from fmeval.data_loaders.data_config import DataConfig
from fmeval.eval_algorithms.save_strategy import SaveStrategy
from fmeval.eval_algorithms.util import get_dataset_configs
from fmeval.eval_algorithms.common import evaluate_dataset
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

RAW_VERDICTS = "raw_verdicts"
STATEMENTS = "statements"
RAW_STATEMENTS = "raw_statements"
LONG_FORM_PROMPT = "long_form_prompt"
NLI_STATEMENTS_PROMT = "nli_statements_prompt"
QUESTION = "question"
ANSWER = "answer"


LONG_FORM_ANSWER_PROMPT = """\
Human: You are given a question and its answer. Your task is to rewrite the answer into one or more simple and coherent statements. Make sure that each statement is faithful to the answer and begins with "Statement:".
Please refer to the following example for the intended output format.
<example>
question: Who was Albert Einstein and what is he best known for?
answer: He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.
statements:
Statement: Albert Einstein was a German-born theoretical physicist.
Statement: Albert Einstein is recognized as one of the greatest and most influential physicists of all time.
Statement: Albert Einstein was best known for developing the theory of relativity.
Statement: Albert Einstein also made important contributions to the development of the theory of quantum mechanics.
</example>

Now solve the following task.
question: $question
answer: $answer
statements:\n Assistant:"""  # noqa: E501


NLI_STATEMENTS_MESSAGE = """
Human: Your task is to judge the faithfulness of a series of statements based on a given context. For each statement you must return verdict as "Yes" if the statement can be directly inferred based on the context or "No" if the statement can not be directly inferred based on the context. In the end, always provide your final verdict for each statement in order. You are given an example below to demonstrate the intended output format.
<example>
context:\nJohn is a student at XYZ University. He is pursuing a degree in Computer Science. He is enrolled in several courses this semester, including Data Structures, Algorithms, and Database Management. John is a diligent student and spends a significant amount of time studying and completing assignments. He often stays late in the library to work on his projects.
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
Final verdicts in order: No. No. Yes. No. Yes.
</example>

Now solve the following task. Remember, you should to judge the faithfulness of the given statements and do not generate your own statements.
context:\n$context
statements:\n$statements
Answer:\n Assistant:"""  # noqa: E501


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
            input_keys=[RAW_VERDICTS, STATEMENTS],
            output_keys=[output_key],
        )
        self.output_key = output_key

    @validate_call
    def __call__(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Augment the input record with the computed faithfulness score.

        :param record: The input record.
        :returns: The input record, with the faithfulness score added in.
        """
        verdict_output = record[RAW_VERDICTS]
        statements = record[STATEMENTS]
        record[self.output_key], error = self._get_score(verdict_output, statements)
        if error:
            record[DatasetColumns.ERROR.value.name] = error
        return record

    @staticmethod
    def _get_score(verdict_output: str, statements: str) -> Tuple[Optional[float], Optional[str]]:
        """Given generated statements and verdicts, compute Faithfulness score.

        :param verdict_output: Verdicts(Yes/No) and explanations string get from Judge model.
        :param statements: Statements string get from `GetStatements` Transform.
        :returns: a tuple of (score, error). Score can range from 0 to 1. See the docstring for `Faithfulness`
            for more details on what these numerical values represent.
        """
        output = verdict_output.lower().strip()
        if statements != "":
            num_statements = len(statements.split("\n"))
            score = float(max(0, output.count("verdict: yes")) / num_statements)
            return score, None
        else:
            return None, "No statements were generated from the answer."


class GetStatements(Transform):
    """This transform invokes a judge model to obtain statements."""

    def __init__(self, input_key: str, output_key: str, judge_model: ModelRunner):
        """GetStatements Initializer.
        :param input_key: The key corresponding to prompt to get statements.
        :param output_key: The key corresponding to the statements, which gets added to the record.
        :param judge_model: An instance of ModelRunner representing the judge model to be used.
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
        """Invokes the judge model to generate statements and augments the input record with the statements.
        :param record: The input record.
        :returns: The input record with the statements added in.
        """
        get_raw_statements = GetModelOutputs(
            input_to_output_keys={self.input_key: [RAW_STATEMENTS]},
            model_runner=self.judge_model,
        )
        record_with_raw_statements = get_raw_statements(record)
        raw_statements = record_with_raw_statements[RAW_STATEMENTS].split("\n")
        list_statements = [statement for statement in raw_statements if statement.startswith("Statement:")]
        statements_str: str = "\n".join([f"{i + 1}. {st}" for i, st in enumerate(list_statements)])
        record[self.output_key] = statements_str
        return record


class Faithfulness(EvalAlgorithmInterface):
    """
    This evaluation measures the factual consistency of the generated answer against the given context.
    It is calculated from answer and retrieved context. The answer is scaled to (0,1) range. Higher the better.

    The generated answer is regarded as faithful if all the claims that are made in the answer can be inferred from
    the given context. To calculate this a set of claims from the generated answer is first identified.
    Then each one of these claims are cross checked with given context to determine if it can be inferred from given
    context or not.
    """

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
            output_keys=[LONG_FORM_PROMPT],
            prompt_template=long_form_prompt_template,
            placeholder_to_record_key={
                QUESTION: DatasetColumns.MODEL_INPUT.value.name,
                ANSWER: DatasetColumns.MODEL_OUTPUT.value.name,
            },
        )
        get_statements = GetStatements(
            input_key=LONG_FORM_PROMPT,
            output_key=STATEMENTS,
            judge_model=judge_model,
        )
        gen_nli_statements_prompt = GeneratePrompt(
            input_keys=[],
            output_keys=[NLI_STATEMENTS_PROMT],
            prompt_template=nli_statements_prompt_template,
            placeholder_to_record_key={
                "context": DatasetColumns.TARGET_CONTEXT.value.name,
                # first STATEMENTS corresponds to $statements in NLI prompt template,
                # second STATEMENTS corresponds to the key that stores the statements, generated by get_statements transform
                STATEMENTS: STATEMENTS,
            },
        )
        get_raw_verdicts = GetModelOutputs(
            input_to_output_keys={NLI_STATEMENTS_PROMT: [RAW_VERDICTS]},
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
        judge_model: ModelRunner,
        long_form_prompt_template: str = LONG_FORM_ANSWER_PROMPT,
        nli_statements_prompt_template: str = NLI_STATEMENTS_MESSAGE,
    ) -> List[EvalScore]:  # type: ignore[override]
        """Compute the faithfulness score on a single sample.

        :param model_input: The expected responses from the model.
        :param model_output: The output of the model being evaluated.
        :param target_context: The relevant context retrieved from RAG system.
        :param judge_model: An instance of ModelRunner representing the judge model to be used.
        :param long_form_prompt_template: The template used to construct the prompt to inference judge model to generate
            statements.
        :param nli_statements_prompt_template: The template used to construct the prompt to inference judge model to get
            verdicts on whether statements relates to given contexts or not.
        :return: A single-element list containing an EvalScore corresponding to the faithfulness score.
        """
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
        if DatasetColumns.ERROR.value.name in result:
            return [
                EvalScore(name=FAITHFULNESS, value=result[FAITHFULNESS], error=result[DatasetColumns.ERROR.value.name])
            ]
        return [EvalScore(name=FAITHFULNESS, value=result[FAITHFULNESS])]

    def evaluate(
        self,
        judge_model: ModelRunner,
        dataset_config: Optional[DataConfig] = None,
        num_records: int = 100,
        save: bool = False,
        save_strategy: Optional[SaveStrategy] = None,
        long_form_prompt_template: str = LONG_FORM_ANSWER_PROMPT,
        nli_statements_prompt_template: str = NLI_STATEMENTS_MESSAGE,
    ) -> List[EvalOutput]:
        """Compute the faithfulness score on one or more datasets.

        :param judge_model: An instance of ModelRunner representing the judge model to be used.
        :param dataset_config: Configures the single dataset used for evaluation.
            If not provided, evaluations will be run on all of this algorithm's built-in datasets.
        :param num_records: The number of records to be sampled randomly from the input dataset(s)
            used to perform the evaluation(s).
        :param save: If set to true, prompt responses and scores will be saved to a file.
        :param save_strategy: Specifies the strategy to use the save the localized outputs of the evaluations. If not
            specified, it will save it to the path that can be configured by the EVAL_RESULTS_PATH environment variable.
            If that environment variable is also not configured, it will be saved to the default path `/tmp/eval_results/`.
        :param long_form_prompt_template: The template used to construct the prompt to inference judge model to generate
            statements.
        :param nli_statements_prompt_template: The template used to construct the prompt to inference judge model to get
            verdicts on if statements relates to given contexts.

        :return: A list of EvalOutput objects.
        """
        dataset_configs = get_dataset_configs(dataset_config, self.eval_name)
        eval_outputs = []
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
