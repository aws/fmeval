import logging
from dataclasses import dataclass
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
from fmeval.exceptions import EvalAlgorithmClientError
from fmeval.model_runners.composers.composers import PromptComposer
from fmeval.model_runners.model_runner import ModelRunner
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
        model_input_key: str = DatasetColumns.MODEL_INPUT.value.name,
        model_output_key: str = DatasetColumns.MODEL_OUTPUT.value.name,
        target_context_key: str = DatasetColumns.TARGET_CONTEXT.value.name,
        output_key: str = FAITHFULNESS,
    ):
        """FaithfulnessScore initializer.

        :param model_input_key: The record key corresponding to the model input.
        :param model_output_key: The record key corresponding to the model output.
        :param target_context_key: The record key corresponding to the contexts.
        :param output_key: The key corresponding to the factual knowledge score that
            will be added to the input record.
        """
        super().__init__(model_input_key, model_output_key, target_context_key, output_key)
        self.register_input_output_keys(
            input_keys=[model_input_key, model_output_key, target_context_key],
            output_keys=[output_key],
        )
        self.model_input_key = model_input_key
        self.model_output_key = model_output_key
        self.target_context_key = target_context_key
        self.output_key = output_key

    @validate_call
    def __call__(self, record: Dict[str, Any], judge_model: ModelRunner) -> Dict[str, Any]:
        """Augment the input record with the computed factual knowledge score.

        :param record: The input record.
        :returns: The input record, with the factual knowledge score added in.
        """
        model_input = record[self.model_input_key]
        model_output = record[self.model_output_key]
        target_context = record[self.target_context_key]
        record[self.output_key] = self._get_score(model_input, model_output, target_context, judge_model)
        return record

    def _get_score(self, model_input: str, model_output: str, target_context: List[str], judge_model: ModelRunner) -> int:
        """Compute the factual knowledge score for a target output and model output pair.

        :param model_input: Model input.
        :param model_output: Model output.
        :param target_context: target context
        :returns: 0 to 1. See the docstring for `Faithfulness` for more details
            on what these numerical values represent.
        """
        human_prompt_composer = PromptComposer(LONG_FORM_ANSWER_PROMPT)
        human_prompt = human_prompt_composer.compose(placeholder_data_dict={"question": model_input, "answer": model_output})
        # step 1: get statement (untested)
        statements_output = judge_model.predict(human_prompt)[0]
        print(statements_output)
        raw_statements = statements_output.split("\n")
        list_statements = [statement for statement in raw_statements if statement.startswith("Statement:")]
        # step 2: generate statement prompt (untested)
        nli_statements_composer = PromptComposer(NLI_STATEMENTS_MESSAGE)
        # statements = "Einstein was born in Germany.\nEinstein was born on 20th March 1879."
        statements_str: str = "\n".join(
            [f"{i + 1}.{st}" for i, st in enumerate(list_statements)]
        )
        print("statement_str:")
        print(statements_str)
        nli_statements_message = nli_statements_composer.compose(placeholder_data_dict={"context": target_context, "statements": statements_str})

        # step 3: get result
        output = judge_model.predict(nli_statements_message)[0]
        output = output.lower().strip()
        print(output)

        # step 4: calculate score
        final_answer = "Final verdicts in order:"
        final_answer = final_answer.lower()
        if output.find(final_answer) != -1:
            print("here")
            output = output[output.find(final_answer) + len(final_answer) :]
            print(output)
            score = sum(
                0 if "yes" in answer else 1
                for answer in output.strip().split(".")
                if answer != ""
            )
            print(score)
            score = score / len(list_statements)
        else:
            score = max(0, output.count("verdict: no")) / len(
                list_statements
            )

        return score


class Faithfulness(EvalAlgorithmInterface):
    """
    """

    eval_name = EvalAlgorithm.FAITHFULNESS.value

    def __init__(self):
        """Faithfulness initializer.
        """
        super().__init__(EvalAlgorithmConfig())
        self.pipeline = TransformPipeline(
            [FaithfulnessScore()]
        )

    # def _build_pipeline(
    #         self,
    #         judge_model: ModelRunner,
    #         prompt_template: str,
    # ) -> TransformPipeline:
    #     transforms = [
    #         get_statements(getmodeloutputtransform), // predict
    #         get_results(getmodeloutputtransform), // use statements, predict
    #         get_score
    #     ]
    #     pipeline = TransformPipeline(transforms)
    #     return pipeline

    def evaluate_sample(self, model_input: str, model_output: str, target_context: str, judge_model: Optional[ModelRunner]=None) -> List[EvalScore]:  # type: ignore[override]
        """Compute the factual knowledge score on a single sample.

        :param model_input: The expected responses from the model.
        :param model_output: The output of the model being evaluated.
        :param target_context: The relevant context retrieved from RAG system.
        :param judge_model: The judge model to be used.
        :return: A single-element list containing an EvalScore corresponding to the faithfulness score.
        """
        if not judge_model:
            judge_model = get_default_judge_model()

        sample = {
            DatasetColumns.MODEL_INPUT.value.name: model_input,
            DatasetColumns.MODEL_OUTPUT.value.name: model_output,
            DatasetColumns.TARGET_CONTEXT.value.name: target_context,
        }
        result = self.pipeline.execute_record(sample, judge_model)
        return [EvalScore(name=FAITHFULNESS, value=result[FAITHFULNESS])]

    def evaluate(
        self,
        judge_model: Optional[ModelRunner] = None,
        dataset_config: Optional[DataConfig] = None,
        prompt_template: Optional[str] = None,
        num_records: int = 300,
        save: bool = False,
        save_strategy: Optional[SaveStrategy] = None,
    ) -> List[EvalOutput]:
        """Compute the faithfulness score on one or more datasets.

        :param model: An instance of ModelRunner representing the model under evaluation.
            If this argument is None, the `dataset_config` argument must not be None,
            and must correspond to a dataset that already contains a column with model outputs.
        :param dataset_config: Configures the single dataset used for evaluation.
            If not provided, evaluations will be run on all of this algorithm's built-in datasets.
        :param prompt_template: A template used to generate prompts that are fed to the model.
            If not provided, defaults will be used. If provided, `model` must not be None.
        :param num_records: The number of records to be sampled randomly from the input dataset(s)
            used to perform the evaluation(s). Note that the default value is 300, rather than
            100, as it is for the rest of the built-in algorithms. This is because there
            are 15 categories for factual knowledge, and if only 100 samples are used, there
            will be categories with very few samples.
        :param save: If set to true, prompt responses and scores will be saved to a file.
        :param save_strategy: Specifies the strategy to use the save the localized outputs of the evaluations. If not
            specified, it will save it to the path that can be configured by the EVAL_RESULTS_PATH environment variable.
            If that environment variable is also not configured, it will be saved to the default path `/tmp/eval_results/`.

        :return: A list of EvalOutput objects.
        """
        dataset_configs = get_dataset_configs(dataset_config, self.eval_name)
        eval_outputs = []
        for dataset_config in dataset_configs:
            dataset = get_dataset(dataset_config, num_records)
            validate_dataset(dataset, [DatasetColumns.MODEL_INPUT.value.name, DatasetColumns.MODEL_OUTPUT.value.name, DatasetColumns.TARGET_CONTEXT.value.name])
            eval_output = evaluate_dataset(
                dataset=dataset,
                pipeline=self.pipeline,
                dataset_name=dataset_config.dataset_name,
                eval_name=self.eval_name,
                metric_names=[FAITHFULNESS],
                eval_results_path=get_eval_results_path(),
                judge_model=judge_model,
                prompt_template=prompt_template,
                agg_method=MEAN,
                save=save,
            )
            eval_outputs.append(eval_output)
        return eval_outputs
