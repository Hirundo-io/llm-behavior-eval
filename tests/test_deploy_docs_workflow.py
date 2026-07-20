from pathlib import Path


def test_docs_deployment_requires_merged_release_pr() -> None:
    workflow = Path(".github/workflows/deploy-docs.yaml").read_text()

    assert """  deploy-docs:
    if: >-
      github.event.pull_request.merged == true &&
      contains(github.event.pull_request.labels.*.name, 'release')
""" in workflow
    assert 'pip install "llm-behavior-eval==${VERSION}"' in workflow
    assert "test.pypi.org" not in workflow
