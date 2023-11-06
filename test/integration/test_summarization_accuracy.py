import os
from pytest import approx
from amazon_fmeval.eval_algorithms.summarization_accuracy import (
    SummarizationAccuracy,
    METEOR_SCORE,
    ROUGE_SCORE,
    BERT_SCORE,
)
from amazon_fmeval.data_loaders.data_config import DataConfig
from amazon_fmeval.constants import MIME_TYPE_JSONLINES
from test.integration.models.model_runners import bedrock_model_runner


ABS_TOL = 5e-3  # Bedrock models are not deterministic, so we use a higher tolerance here
os.environ["PARALLELIZATION_FACTOR"] = "2"
eval_algo = SummarizationAccuracy()


def format_input(input_str: str) -> str:
    """
    Formats the input to match what is required by Claude,
    specifically, anthropic.claude-v2.
    """
    return f"Human: {input_str}\n\nAssistant:\n"


class TestSummarizationAccuracy:
    def test_evaluate_sample(self):
        original_text = 'The full cost of damage in Newton Stewart, one of the areas worst affected, is still being assessed.\nRepair work is ongoing in Hawick and many roads in Peeblesshire remain badly affected by standing water.\nTrains on the west coast mainline face disruption due to damage at the Lamington Viaduct.\nMany businesses and householders were affected by flooding in Newton Stewart after the River Cree overflowed into the town.\nFirst Minister Nicola Sturgeon visited the area to inspect the damage.\nThe waters breached a retaining wall, flooding many commercial properties on Victoria Street - the main shopping thoroughfare.\nJeanette Tate, who owns the Cinnamon Cafe which was badly affected, said she could not fault the multi-agency response once the flood hit.\nHowever, she said more preventative work could have been carried out to ensure the retaining wall did not fail.\n"It is difficult but I do think there is so much publicity for Dumfries and the Nith - and I totally appreciate that - but it is almost like we\'re neglected or forgotten," she said.\n"That may not be true but it is perhaps my perspective over the last few days.\n"Why were you not ready to help us a bit more when the warning and the alarm alerts had gone out?"\nMeanwhile, a flood alert remains in place across the Borders because of the constant rain.\nPeebles was badly hit by problems, sparking calls to introduce more defences in the area.\nScottish Borders Council has put a list on its website of the roads worst affected and drivers have been urged not to ignore closure signs.\nThe Labour Party\'s deputy Scottish leader Alex Rowley was in Hawick on Monday to see the situation first hand.\nHe said it was important to get the flood protection plan right but backed calls to speed up the process.\n"I was quite taken aback by the amount of damage that has been done," he said.\n"Obviously it is heart-breaking for people who have been forced out of their homes and the impact on businesses."\nHe said it was important that "immediate steps" were taken to protect the areas most vulnerable and a clear timetable put in place for flood prevention plans.\nHave you been affected by flooding in Dumfries and Galloway or the Borders? Tell us about your experience of the situation and how it was handled. Email us on selkirk.news@bbc.co.uk or dumfries@bbc.co.uk.'
        model_input = f"Summarise the following text in one sentence: {original_text}"
        target_output = "Clean-up operations are continuing across the Scottish Borders and Dumfries and Galloway after flooding caused by Storm Frank."
        model_output = bedrock_model_runner.predict(format_input(model_input))[0]
        eval_scores = eval_algo.evaluate_sample(target_output, model_output)
        for eval_score in eval_scores:  # pragma: no branch
            if eval_score.name == METEOR_SCORE:
                assert eval_score.value == approx(0.380, abs=ABS_TOL)
            elif eval_score.name == ROUGE_SCORE:
                assert eval_score.value == approx(0.250, abs=ABS_TOL)
            elif eval_score.name == BERT_SCORE:
                assert eval_score.value == approx(0.734, abs=ABS_TOL)

    def test_evaluate(self, integration_tests_dir):
        dataset_config = DataConfig(
            dataset_name="xsum_sample",
            dataset_uri=os.path.join(integration_tests_dir, "datasets", "xsum_sample.jsonl"),
            dataset_mime_type=MIME_TYPE_JSONLINES,
            model_input_location="document",
            target_output_location="summary",
        )
        eval_outputs = eval_algo.evaluate(
            model=bedrock_model_runner,
            dataset_config=dataset_config,
            prompt_template=format_input("Summarise the following text in one sentence: $feature"),
            save=True,
        )
        eval_output = eval_outputs[0]
        for eval_score in eval_output.dataset_scores:
            if eval_score.name == METEOR_SCORE:
                assert eval_score.value == approx(0.279, abs=ABS_TOL)
            elif eval_score.name == ROUGE_SCORE:
                assert eval_score.value == approx(0.084, abs=ABS_TOL)
            elif eval_score.name == BERT_SCORE:
                assert eval_score.value == approx(0.677, abs=ABS_TOL)
