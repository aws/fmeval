import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union

import numpy as np

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
from fmeval.eval_algorithms.util import validate_dataset, validate_prompt_template
from fmeval.model_runners.model_runner import ModelRunner
from fmeval.transforms.common import GetModelOutputs, GeneratePrompt
from fmeval.transforms.transform import Transform
from fmeval.transforms.transform_pipeline import TransformPipeline
from fmeval.transforms.util import validate_call
from fmeval.util import get_eval_results_path

logger = logging.getLogger(__name__)

ANSWER_RELEVANCE = EvalAlgorithm.ANSWER_RELEVANCE.value

QUESTION_GENERATION_PROMPT = "question_generation_prompt"
ANSWER = "answer"
RAW_GEN_QUESTIONS = "raw_gen_questions"
GEN_QUESTIONS = "gen_questions"

QUESTION_GEN_PROMPT = """\
Human: Generate $strictness question(s) for the given answer. Make sure that each question begins with "Question:".
Please refer to the following example for the intended output format.
<example>
Answer:\nThe PSLV-C56 mission is scheduled to be launched on Sunday, 30 July 2023 at 06:30 IST / 01:00 UTC. It will be launched from the Satish Dhawan Space Centre, Sriharikota, Andhra Pradesh, India
Question: When is the scheduled launch date and time for the PSLV-C56 mission, and where will it be launched from?
<example>

Now solve the following task.
Answer:$answer
Question:Assistant:"""  # noqa: E501


@dataclass(frozen=True)
class AnswerRelevanceConfig(EvalAlgorithmConfig):
    """Configures the Answer Relevance evaluation algorithm.

    :param strictness: The number questions generated per answer.
    """

    strictness: Optional[int] = 1


class AnswerRelevanceScore(Transform):
    """This transform augments its input record with the computed answer relevance score.

    See the docstring for `Answer Relevance` for more details regarding the score itself.
    """

    def __init__(
        self,
        embeddings_model: ModelRunner,
        model_output_key: str = DatasetColumns.MODEL_OUTPUT.value.name,
        output_key: str = ANSWER_RELEVANCE,
    ):
        """AnswerRelevanceScore initializer.

        :param embeddings_model: An instance of ModelRunner representing the embedding model to be used.
        :param model_output_key: The key corresponding to the model output.
        :param output_key: The key corresponding to the answer relevance score that will be added to the input record.
        """
        super().__init__(embeddings_model, model_output_key, output_key)
        self.register_input_output_keys(
            input_keys=[GEN_QUESTIONS, model_output_key],
            output_keys=[output_key],
        )
        self.output_key = output_key
        self.model_output_key = model_output_key
        self.embeddings_model = embeddings_model

    @validate_call
    def __call__(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Augment the input record with the computed answer relevance score.

        :param record: The input record.
        :returns: The input record, with the answer relevance score added in.
        """
        gen_questions_str = record[GEN_QUESTIONS]
        question = record[self.model_output_key]
        record[self.output_key] = self._get_score(question, gen_questions_str)
        return record

    def _get_score(self, question: str, gen_questions_str: str) -> float:
        """Given generated questions and the original question, embedding model, computer Answer Relevance score.

        :param question: The given prompt (model input).
        :param gen_questions_str: Questions generated by judge model based on the answer.
        :returns: 0 to 1. See the docstring for `Answer Relevance` for more details
            on what these numerical values represent.
        """
        question_vector = np.asarray(self.embeddings_model.predict(question))
        question_norm = np.linalg.norm(question_vector)
        total = 0
        if gen_questions_str:
            generated_questions = gen_questions_str.split("\n")
            for generated_question in generated_questions:
                gen_question_vector = np.asarray(self.embeddings_model.predict(generated_question))
                norm = np.linalg.norm(gen_question_vector) * question_norm
                total += np.dot(gen_question_vector, question_vector.T) / norm
            return total / len(generated_questions)
        # no questions being generated
        return 0


class GenerateQuestions(Transform):
    """This transform invokes a judge model to generate questions."""

    def __init__(self, input_key: str, output_key: str):
        """GenerateQuestions Initializer.
        :param input_key: The key corresponding to prompt to get statements.
        :param output_key: The key corresponding to the statements, which gets added to the record.
        """
        super().__init__(input_key, output_key)
        self.register_input_output_keys(
            input_keys=[input_key],
            output_keys=[output_key],
        )
        self.input_key = input_key
        self.output_key = output_key

    @validate_call
    def __call__(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Invokes the judge model to generate statements and augments the input record with the statements.
        :param record: The input record.
        :returns: The input record with the statements added in.
        """
        raw_questions = record[RAW_GEN_QUESTIONS].split("\n")
        list_questions = [question.strip() for question in raw_questions if question.strip().startswith("Question: ")]
        questions_str: str = "\n".join([question.removeprefix("Question: ") for question in list_questions])
        record[self.output_key] = questions_str
        return record


class AnswerRelevance(EvalAlgorithmInterface):
    """
    The evaluation metric, Answer Relevancy, focuses on assessing how pertinent the generated answer is to the
    given prompt. It is defined as the mean cosine similarity of the original question to a number of artifical
    questions, which where generated (reverse engineered) based on the answer.
    A lower score is assigned to answers that are incomplete or contain redundant information and higher scores
    indicate better relevancy.
    """

    eval_name = EvalAlgorithm.ANSWER_RELEVANCE.value

    def __init__(self, eval_algorithm_config: AnswerRelevanceConfig = AnswerRelevanceConfig()):
        """Default constructor

        :param eval_algorithm_config: Answer Relevance eval algorithm config.
        """
        super().__init__(eval_algorithm_config)
        self.strictness = eval_algorithm_config.strictness

    @staticmethod
    def _build_pipeline(
        judge_model: ModelRunner,
        embeddings_model: ModelRunner,
        question_generation_prompt_template: str,
    ) -> TransformPipeline:
        gen_question_generation_prompt = GeneratePrompt(
            input_keys=[],
            output_keys=[QUESTION_GENERATION_PROMPT],
            prompt_template=question_generation_prompt_template,
            placeholder_to_record_key={
                ANSWER: DatasetColumns.MODEL_OUTPUT.value.name,
            },
        )
        gen_raw_questions = GetModelOutputs(
            input_to_output_keys={QUESTION_GENERATION_PROMPT: [RAW_GEN_QUESTIONS]},
            model_runner=judge_model,
        )
        gen_questions = GenerateQuestions(
            input_key=RAW_GEN_QUESTIONS,
            output_key=GEN_QUESTIONS,
        )
        compute_score = AnswerRelevanceScore(embeddings_model=embeddings_model)
        transforms = [gen_question_generation_prompt, gen_raw_questions, gen_questions, compute_score]
        pipeline = TransformPipeline(transforms)
        return pipeline

    def evaluate_sample(
        self,
        model_input: str,
        model_output: str,
        judge_model: ModelRunner,
        embeddings_model: ModelRunner,
        question_generation_prompt_template: str = QUESTION_GEN_PROMPT,
    ) -> List[EvalScore]:  # type: ignore[override]
        """Compute the Answer Relevance score on a single sample.

        :param model_input: The expected responses from the model.
        :param model_output: The output of the model being evaluated.
        :param judge_model: An instance of ModelRunner representing the judge model to be used.
        :param embeddings_model: An instance of ModelRunner representing the embedding model to be used.
        :param question_generation_prompt_template: The template used to construct the prompt to inference
            judge model to get generate question based on answer. Must contain $strictness and $answer keyword
            placeholders.
        :return: A single-element list containing an EvalScore corresponding to the answer relevance score.
        """
        validate_prompt_template(question_generation_prompt_template, ["strictness", "answer"])
        question_generation_prompt_template = question_generation_prompt_template.replace(
            "$strictness", str(self.strictness)
        )
        sample = {
            DatasetColumns.MODEL_INPUT.value.name: model_input,
            DatasetColumns.MODEL_OUTPUT.value.name: model_output,
        }
        pipeline = self._build_pipeline(
            judge_model=judge_model,
            question_generation_prompt_template=question_generation_prompt_template,
            embeddings_model=embeddings_model,
        )
        result = pipeline.execute_record(sample)
        return [EvalScore(name=ANSWER_RELEVANCE, value=result[ANSWER_RELEVANCE])]

    def evaluate(
        self,
        judge_model: ModelRunner,
        embeddings_model: ModelRunner,
        dataset_config: Optional[Union[DataConfig, List[DataConfig]]] = None,
        num_records: int = 300,
        save: bool = False,
        save_strategy: Optional[SaveStrategy] = None,
        question_generation_prompt_template: str = QUESTION_GEN_PROMPT,
    ) -> List[EvalOutput]:
        """Compute the answer relevance score on one or more datasets.

        :param judge_model: An instance of ModelRunner representing the judge model to be used.
        :param embeddings_model: An instance of ModelRunner representing the embedding model to be used.
        :param dataset_config: Configures a single dataset or list of datasets used for the
            evaluation. If not provided, this method will run evaluations using all of its
            supported built-in datasets.
        :param num_records: The number of records to be sampled randomly from the input dataset(s)
            used to perform the evaluation(s). Note that the default value is 300, rather than
            100, as it is for the rest of the built-in algorithms.
        :param save: If set to true, prompt responses and scores will be saved to a file.
        :param save_strategy: Specifies the strategy to use the save the localized outputs of the evaluations. If not
            specified, it will save it to the path that can be configured by the EVAL_RESULTS_PATH environment variable.
            If that environment variable is also not configured, it will be saved to the default path `/tmp/eval_results/`.
        :param question_generation_prompt_template: The template used to construct the prompt to inference judge model
            to get generate question based on answer. Must contain $strictness and $answer keyword placeholders.

        :return: A list of EvalOutput objects.
        """
        validate_prompt_template(question_generation_prompt_template, ["strictness", "answer"])
        question_generation_prompt_template = question_generation_prompt_template.replace(
            "$strictness", str(self.strictness)
        )
        dataset_configs = get_dataset_configs(dataset_config, self.eval_name)
        eval_outputs = []
        for dataset_config in dataset_configs:
            dataset = get_dataset(dataset_config, num_records)
            validate_dataset(
                dataset,
                [
                    DatasetColumns.MODEL_INPUT.value.name,
                    DatasetColumns.MODEL_OUTPUT.value.name,
                ],
            )
            eval_output = evaluate_dataset(
                dataset=dataset,
                pipeline=self._build_pipeline(judge_model, embeddings_model, question_generation_prompt_template),
                dataset_name=dataset_config.dataset_name,
                eval_name=self.eval_name,
                metric_names=[ANSWER_RELEVANCE],
                eval_results_path=get_eval_results_path(),
                agg_method=MEAN,
                save=save,
                save_strategy=save_strategy,
            )
            eval_outputs.append(eval_output)
        return eval_outputs
