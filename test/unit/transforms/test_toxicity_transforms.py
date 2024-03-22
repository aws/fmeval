from typing import List, Dict

import numpy as np

from fmeval.helper_models import ToxigenModel
from fmeval.transforms.toxicity import ToxicityScores


class TestDetector(ToxigenModel):
    """Toxigen helper model.

    See https://huggingface.co/tomh/toxigen_roberta/tree/main
    """

    SCORE_NAMES = ["toxicity", "hate"]

    # noinspection PyMissingConstructor
    def __init__(self):
        pass

    def invoke_model(self, text_inputs: List[str]) -> Dict[str, List[float]]:
        """Get Toxigen scores by invoking self._model on a list of text inputs.

        Note: Toxigen scores are for the label "LABEL_1".

        :param text_inputs: A list of text inputs for the model.
        :returns: A dict mapping score name to a list of scores for each of the text inputs.
        """
        return {
            TestDetector.SCORE_NAMES[0]: [0.1] * len(text_inputs),
            TestDetector.SCORE_NAMES[1]: [0.9] * len(text_inputs),
        }


def test_detector_score_call():
    """
    GIVEN a ToxicityScores instance.
    WHEN its __call__ method is invoked.
    THEN the results are as expected given the TestDetector above.

    Note: we don't validate the structure of the __call__ output since
    we already have @validate_call to handle that.
    """
    sample = {"target_output": ["Hello there!", "yess"], "model_output": ["Hi", "sir"]}
    model = TestDetector()
    toxicity_scores = ToxicityScores(
        ["target_output", "model_output"], score_names=model.SCORE_NAMES, toxicity_detector=model
    )
    result = toxicity_scores(sample)
    assert np.allclose(result["target_output_toxicity"], [0.1, 0.1])
    assert np.allclose(result["model_output_hate"], [0.9, 0.9])
