import os
from testbook import testbook
from testbook.client import TestbookNotebookClient

from fmeval.util import project_root


bedrock_example_notebook_path = os.path.join(
    project_root(__file__), "examples", "bedrock-claude-factual-knowledge.ipynb"
)


@testbook(bedrock_example_notebook_path, timeout=600)
def test_bedrock_model_notebook(tb):
    tb.inject(
        """
        import json
        from unittest.mock import patch, MagicMock, mock_open
        from io import StringIO
        from botocore.response import StreamingBody
        mock_bedrock = MagicMock()
        body_encoded = '{"completion": "some text"}'
        mock_bedrock.invoke_model.return_value = {"body": StreamingBody(StringIO(body_encoded), len(body_encoded))}
        p1 = patch('boto3.client', return_value=mock_bedrock)
        p1.start()
        mock_algo = MagicMock()
        mock_algo.evaluate.return_value = []
        p2 = patch('fmeval.eval_algorithms.factual_knowledge.FactualKnowledge', return_value=mock_algo)
        p2.start()
        mock_br_model_runner = MagicMock()
        p3 = patch('fmeval.model_runners.bedrock_model_runner.BedrockModelRunner', return_value=mock_br_model_runner)
        p3.start()
        data = {"scores": [{'name': 'factual_knowledge', 'value': 0}]}
        # This is the equivalent of patching 'builtins.open' but for notebooks
        p4 = patch('IPython.core.interactiveshell.io_open', mock_open(read_data=json.dumps(data)))
        p4.start()
        """
    )

    # Skip execution of the first code cell, which installs the *currently-released* fmeval package.
    # We want to test the example notebooks against the package built from the latest code
    # (see devtool build_package and install_package), not the released package.
    seen_code_cell = False
    for index, cell in enumerate(tb.cells):
        if cell["cell_type"] == "code" and not seen_code_cell:
            seen_code_cell = True
            continue
        super(TestbookNotebookClient, tb).execute_cell(cell, index)

    tb.inject(
        """
        p1.stop()
        p2.stop()
        p3.stop()
        p4.stop()
        """
    )


js_model_example_notebook_path = os.path.join(project_root(__file__), "examples", "jumpstart-falcon-stereotyping.ipynb")


@testbook(js_model_example_notebook_path, timeout=600)
def test_js_model_notebook(tb):
    tb.inject(
        """
        import json
        from unittest.mock import patch, MagicMock, mock_open
        js_model = MagicMock()
        mock_predictor = MagicMock()
        js_model.deploy.return_value = mock_predictor
        p1 = patch('sagemaker.jumpstart.model.JumpStartModel', return_value=js_model)
        p1.start()
        mock_js_model_runner = MagicMock()
        p2 = patch('fmeval.model_runners.sm_jumpstart_model_runner.JumpStartModelRunner', return_value=mock_js_model_runner)
        p2.start()
        mock_algo = MagicMock()
        mock_algo.evaluate.return_value = []
        p3 = patch('fmeval.eval_algorithms.prompt_stereotyping.PromptStereotyping', return_value=mock_algo)
        p3.start()
        data = {"scores": [{'name': 'prompt_stereotyping', 'value': 0}]}
        # This is the equivalent of patching 'builtins.open' but for notebooks
        p4 = patch('IPython.core.interactiveshell.io_open', mock_open(read_data=json.dumps(data)))
        p4.start()
        """
    )

    # Skip execution of the first code cell, which installs the *currently-released* fmeval package.
    # We want to test the example notebooks against the package built from the latest code
    # (see devtool build_package and install_package), not the released package.
    seen_code_cell = False
    for index, cell in enumerate(tb.cells):
        if cell["cell_type"] == "code" and not seen_code_cell:
            seen_code_cell = True
            continue
        super(TestbookNotebookClient, tb).execute_cell(cell, index)

    tb.inject(
        """
        p1.stop()
        p2.stop()
        p3.stop()
        p4.stop()
        """
    )


custom_model_chatgpt_example_notebook_path = os.path.join(
    project_root(__file__), "examples", "custom_model_runner_chat_gpt.ipynb"
)


@testbook(custom_model_chatgpt_example_notebook_path, timeout=600)
def test_custom_model_chat_gpt_notebook(tb):
    tb.inject(
        """
        import json
        from unittest.mock import patch, MagicMock, mock_open
        from requests.models import Response
        mock_response = Response()
        mock_response.status_code = 200
        mock_response._content = str.encode('{"choices": [{"message": {"content": "text"}}]}')
        p1 = patch('requests.request', return_value=mock_response)
        p1.start()
        mock_algo = MagicMock()
        mock_algo.evaluate.return_value = []
        p2 = patch('fmeval.eval_algorithms.factual_knowledge.FactualKnowledge', return_value=mock_algo)
        p2.start()
        data = {"scores": [{'name': 'factual_knowledge', 'value': 0}]}
        # This is the equivalent of patching 'builtins.open' but for notebooks
        p3 = patch('IPython.core.interactiveshell.io_open', mock_open(read_data=json.dumps(data)))
        p3.start()
        """
    )

    # Skip execution of the first code cell, which installs the *currently-released* fmeval package.
    # We want to test the example notebooks against the package built from the latest code
    # (see devtool build_package and install_package), not the released package.
    seen_code_cell = False
    for index, cell in enumerate(tb.cells):
        if cell["cell_type"] == "code" and not seen_code_cell:
            seen_code_cell = True
            continue
        super(TestbookNotebookClient, tb).execute_cell(cell, index)

    tb.inject(
        """
        p1.stop()
        p2.stop()
        p3.stop()
        """
    )


custom_model_hf_example_notebook_path = os.path.join(project_root(__file__), "examples", "custom_model_runner_hf.ipynb")


@testbook(custom_model_hf_example_notebook_path, timeout=600)
def test_custom_model_hf_notebook(tb):
    tb.inject(
        """
        import json
        import torch
        from unittest.mock import patch, MagicMock, mock_open
        mock_algo = MagicMock()
        mock_algo.evaluate.return_value = []
        p1 = patch('fmeval.eval_algorithms.factual_knowledge.FactualKnowledge', return_value=mock_algo)
        p1.start()

        mock_model = MagicMock()
        mock_model.generate.return_value = torch.tensor([[50256, 198, 464, 717, 640, 314, 2497, 262, 649, 2196, 286,
            262, 983, 11, 314, 373, 523, 6568, 13, 314, 373, 523, 6568, 284, 766, 262, 649, 2196, 286, 262, 983, 11,
            314]])
        p2 = patch('transformers.AutoModelForCausalLM.from_pretrained', return_value=mock_model)
        p2.start()

        mock_tokenizer = MagicMock()
        mock_tokenizer().to.return_value = {"input_ids": torch.tensor([[23421, 318, 262, 3139, 286, 30]])}
        p3 = patch('transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer)
        p3.start()

        data = {"scores": [{'name': 'factual_knowledge', 'value': 0}]}
        # This is the equivalent of patching 'builtins.open' but for notebooks
        p4 = patch('IPython.core.interactiveshell.io_open', mock_open(read_data=json.dumps(data)))
        p4.start()
        """
    )

    # Skip execution of the first code cell, which installs the *currently-released* fmeval package.
    # We want to test the example notebooks against the package built from the latest code
    # (see devtool build_package and install_package), not the released package.
    seen_code_cell = False
    for index, cell in enumerate(tb.cells):
        if cell["cell_type"] == "code" and not seen_code_cell:
            seen_code_cell = True
            continue
        super(TestbookNotebookClient, tb).execute_cell(cell, index)

    tb.inject(
        """
        p1.stop()
        p2.stop()
        p3.stop()
        p4.stop()
        """
    )
