import pytest
import ray

from fmeval.helper_models import BertscoreModel, BertscoreModelTypes
from fmeval.util import create_shared_resource


class TestUtil:
    def test_create_shared_resource(self):
        """
        GIVEN a BertscoreModel instance.
        WHEN create_shared_resource is called on this instance.
        THEN a Ray actor handle for the BertscoreModel is returned,
            and this actor handle can be used just like a regular
            BertscoreModel object (with the addition of needing to call
            `remote` and `get`).

        Note that the input payload and expected result are copied from
        the BertscoreModel.invoke_model unit test.
        """
        bertscore_model = BertscoreModel(BertscoreModelTypes.ROBERTA_MODEL.value)
        actor_handle = create_shared_resource(bertscore_model)
        result = ray.get(actor_handle.invoke_model.remote("sample text reference", "sample text prediction"))
        assert result == pytest.approx(0.8580247163772583)
