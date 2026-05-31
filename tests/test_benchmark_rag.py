from app.api import ChunkIn
from benchmarks.benchmark_rag import build_answer_prompt, run_case
from benchmarks.metrics import BenchmarkCase


def test_build_answer_prompt_contains_question_and_context() -> None:
    prompt = build_answer_prompt("What is retrieval?", ["Passage one.", "Passage two."])

    assert "What is retrieval?" in prompt
    assert "[1] Passage one." in prompt
    assert "[2] Passage two." in prompt
    assert "using only the provided context" in prompt


def test_run_case_smoke_with_fake_clients(monkeypatch) -> None:
    case = BenchmarkCase(
        id="case-1",
        question="What is tested?",
        expected_answer="SlimContext is tested.",
        chunks=["SlimContext is tested with fake clients.", "Other retrieval context."],
    )

    def fake_embed_case(case, embedding_model_name):
        return (
            [
                ChunkIn(id="a", text=case.chunks[0], embedding=[1.0, 0.0], score=0.9),
                ChunkIn(id="b", text=case.chunks[1], embedding=[0.0, 1.0], score=0.5),
            ],
            [1.0, 0.0],
        )

    class FakeAnswerClient:
        def answer(self, model, question, context):
            return "SlimContext is tested.", 10.0

    class FakeJudge:
        def score(self, question, expected_answer, actual_answer, context):
            return 8.0

    monkeypatch.setattr("benchmarks.benchmark_rag.embed_case", fake_embed_case)

    result = run_case(
        case=case,
        answer_client=FakeAnswerClient(),
        judge=FakeJudge(),
        model="fake-model",
        embedding_model_name="fake-embedding-model",
        target_k=1,
        token_budget=100,
    )

    assert result.id == "case-1"
    assert result.baseline_input_tokens >= result.slimcontext_input_tokens
    assert result.baseline_quality == 8.0
    assert result.slimcontext_quality == 8.0
