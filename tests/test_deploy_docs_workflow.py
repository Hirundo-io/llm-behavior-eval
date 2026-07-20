from pathlib import Path

import yaml


def test_docs_deployment_requires_merged_release_pr() -> None:
    workflow_text = Path(".github/workflows/deploy-docs.yaml").read_text()
    workflow = yaml.safe_load(workflow_text)
    condition = workflow["jobs"]["deploy-docs"]["if"]

    assert "github.event.pull_request.merged == true" in condition
    assert "contains(github.event.pull_request.labels.*.name, 'release')" in condition
    assert 'pip install "llm-behavior-eval==${VERSION}"' in workflow_text
    assert "test.pypi.org" not in workflow_text
