import pytest

from fmeval.eval_algorithms.summarization_accuracy import SummarizationAccuracyConfig, SummarizationAccuracy


class TestStereotypingEvaluation:

    DATA_WITH_CORRECT_VALUES = [
        {  # fully correct; scores should be (close to) 1.
            "article": "Cake is so delicious, I really like cake. I want to open a bakery when I grow up.",
            "target_summary": "I like cake very much.",
            "model_summary": "I like cake very much.",
            "rouge1": 1.0,
            "meteor": 0.99769,  # this is not == 1 due to brevity penalty (gamma parameter)
            "bert_score_default_model": 1.0,
        },
        {  # somewhat correct; scores should be > 0 and < 1
            "article": "The art metropolis of Berlin inspires locals and visitors with its famous "
            "museum landscape and numerous UNESCO World Heritage sites."
            " It is also an international exhibition venue. "
            "You will find a selection of current and upcoming exhibitions here.",
            "target_summary": "Berlin an art metropolis",
            "model_summary": "Berlin Art, Heritage, Exhibitions Hub",
            "rouge1": 2 * (2 / 5 * 2 / 4) / (2 / 5 + 2 / 4),  # F1 score over 1-gram matches
            "meteor": 0.232558139,
            "bert_score_default_model": 0.6644857,
        },
        {  # entirely incorrect; scores should be (close to) 0
            "article": "The art metropolis of Berlin inspires locals and visitors with its famous "
            "museum landscape and numerous UNESCO World Heritage sites."
            " It is also an international exhibition venue. "
            "You will find a selection of current and upcoming exhibitions here.",
            "target_summary": "Berlin: an art metropolis.",
            "model_summary": "Who likes brownies?",
            "rouge1": 0.0,
            "meteor": 0.0,
            "bert_score_default_model": 0.537276,  # surprisingly large
        },
    ]

    @pytest.mark.parametrize("data_with_predictions", DATA_WITH_CORRECT_VALUES)
    def test_correctness(self, data_with_predictions):
        """Test that the scores are computed (scientifically) correctly."""
        config = SummarizationAccuracyConfig(rouge_type="rouge1")
        eval_algorithm = SummarizationAccuracy(config)
        responses = eval_algorithm.evaluate_sample(
            data_with_predictions["target_summary"], data_with_predictions["model_summary"]
        )

        # meteor
        assert responses[0].value == pytest.approx(data_with_predictions["meteor"], rel=1e-5)
        # rouge1
        assert responses[1].value == pytest.approx(data_with_predictions["rouge1"], rel=1e-5)
        # bertscore
        assert responses[2].value == pytest.approx(data_with_predictions["bert_score_default_model"], rel=1e-5)
