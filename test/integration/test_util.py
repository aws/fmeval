import pytest
import ray

from fmeval.eval_algorithms.helper_models.helper_model import BertscoreHelperModel, BertscoreHelperModelTypes
from fmeval.util import create_shared_resource


class TestUtil:
    def test_create_shared_resource(self):
        """
        GIVEN a BertscoreHelperModel instance.
        WHEN create_shared_resource is called on this instance.
        THEN a Ray actor handle for the BertscoreHelperModel is returned,
            and this actor handle can be used just like a regular
            BertscoreHelperModel object (with the addition of needing to call
            `remote` and `get`).

        Note that the input payload and expected result are copied from
        the BertscoreHelperModel.get_helper_scores unit test.
        """
        bertscore_model = BertscoreHelperModel(BertscoreHelperModelTypes.ROBERTA_MODEL.value)
        actor_handle = create_shared_resource(bertscore_model)
        result = ray.get(actor_handle.get_helper_scores.remote("sample text reference", "sample text prediction"))
        assert result == pytest.approx(0.8580247163772583)
