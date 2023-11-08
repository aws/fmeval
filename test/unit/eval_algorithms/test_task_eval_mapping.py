from fmeval.eval_algorithms import EvalAlgorithm


def test_eval_mapping():
    assert str(EvalAlgorithm.PROMPT_STEREOTYPING) == "PROMPT STEREOTYPING"
