from fmeval.data_loaders.util import get_dataset
from fmeval.transforms.common import GeneratePrompt, GetModelOutputs
from fmeval.transforms.transform_pipeline import TransformPipeline
from fmeval.eval_algorithms import DATASET_CONFIGS, TREX
from test.integration.models.model_runners import sm_model_runner


def test_pipeline_execution():
    """
    GIVEN a dataset and a TransformPipeline.
    WHEN the pipeline's execute() method is called on the dataset.
    THEN Ray successfully applies the transforms to the dataset.
    """
    data_config = DATASET_CONFIGS[TREX]
    ds = get_dataset(data_config, 20)
    original_columns = set(ds.columns())

    gen_prompt = GeneratePrompt(
        input_keys=["model_input"],
        output_keys=["prompt"],
        prompt_template="Summarize the following text in one sentence: $model_input",
    )

    get_model_output = GetModelOutputs(
        input_to_output_keys={gen_prompt.output_keys[0]: ["model_output"]},
        model_runner=sm_model_runner,
    )

    pipeline = TransformPipeline([gen_prompt, get_model_output])
    ds = pipeline.execute(ds)

    new_columns = set(ds.columns())
    assert new_columns - original_columns == {"prompt", "model_output"}
