"""Tests for the weekly finance newsletter workflow with stubbed external effects."""

import asyncio
from collections.abc import Awaitable, Callable
from datetime import timedelta

import pytest
from mistralai import workflows
from mistralai.client.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ResponseFormat,
)
from mistralai.workflows.testing import create_test_worker
from worker.workflows.finance_newsletter import (
    NewsletterContent,
    Story,
    WeeklyFinanceNewsletterWorkflow,
)

WORKFLOW_EXECUTION_TIMEOUT = timedelta(seconds=30)

STORIES = [
    Story(
        title="Fed holds rates steady",
        url="https://example.com/fed",
        summary="The Federal Reserve kept rates unchanged at 4.5%.",
    ),
    Story(
        title="Markets rally on earnings",
        url="https://example.com/markets",
        summary="The S&P 500 gained 2.1% after strong bank earnings.",
    ),
]

CONTENT = NewsletterContent(
    subject="This week in finance",
    intro="Here is your weekly recap.",
    top_stories=STORIES,
    podcast_script="Hello and welcome to your weekly finance recap.",
    plain_text="This is the plain text newsletter.",
)

FAKE_MP3 = b"ID3fakeaudio"


@workflows.activity(name="fetch-financial-news")
async def stub_fetch_financial_news() -> list[Story]:
    return STORIES


@workflows.activity(name="generate-audio-recap")
async def stub_generate_audio(script: str) -> bytes:
    return FAKE_MP3


@workflows.activity(name="mistralai_chat_parse")
async def stub_mistralai_chat_parse(
    params: ChatCompletionRequest, response_format: ResponseFormat
) -> ChatCompletionResponse:
    from mistralai.client.models import (
        AssistantMessage,
        ChatCompletionChoice,
        UsageInfo,
    )

    return ChatCompletionResponse(
        id="test",
        object="chat.completion",
        model="mistral-medium-latest",
        usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        created=0,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=AssistantMessage(
                    role="assistant", content=CONTENT.model_dump_json()
                ),
                finish_reason="stop",
            )
        ],
    )


@pytest.fixture
def stub_activities() -> list[Callable[..., Awaitable]]:
    return [stub_fetch_financial_news, stub_generate_audio, stub_mistralai_chat_parse]


async def test_dry_run_composes_newsletter_and_audio(
    temporal_env, stub_activities
) -> None:
    async with create_test_worker(
        temporal_env,
        workflows=[WeeklyFinanceNewsletterWorkflow],
        activities=stub_activities,
    ):
        handle = await temporal_env.client.start_workflow(
            "weekly-finance-newsletter",
            {"dry_run": True, "with_audio": True},
            id="test-finance-newsletter-dry-run",
            task_queue="test-task-queue",
            execution_timeout=WORKFLOW_EXECUTION_TIMEOUT,
        )
        result = await asyncio.wait_for(handle.result(), timeout=20)

    assert result["subject"] == CONTENT.subject
    assert result["email_sent"] is False
    assert result["audio_generated"] is True
    assert len(result["newsletter"]["top_stories"]) == 2


async def test_send_delivers_email_with_audio_attachment(
    temporal_env, stub_activities
) -> None:
    sent: list[dict] = []

    @workflows.activity(name="send-newsletter-email")
    async def stub_send_newsletter_email(
        recipient: str,
        subject: str,
        html: str,
        plain_text: str,
        audio_base64: str | None,
    ) -> None:
        sent.append(
            {
                "recipient": recipient,
                "subject": subject,
                "html": html,
                "plain_text": plain_text,
                "audio_base64": audio_base64,
            }
        )

    async with create_test_worker(
        temporal_env,
        workflows=[WeeklyFinanceNewsletterWorkflow],
        activities=[*stub_activities, stub_send_newsletter_email],
    ):
        handle = await temporal_env.client.start_workflow(
            "weekly-finance-newsletter",
            {"recipient": "reader@example.com", "dry_run": False, "with_audio": True},
            id="test-finance-newsletter-send",
            task_queue="test-task-queue",
            execution_timeout=WORKFLOW_EXECUTION_TIMEOUT,
        )
        result = await asyncio.wait_for(handle.result(), timeout=20)

    assert result["email_sent"] is True
    assert result["recipient"] == "reader@example.com"
    assert len(sent) == 1
    assert sent[0]["recipient"] == "reader@example.com"
    assert "weekly-finance-recap.mp3" in sent[0]["html"]
    assert sent[0]["audio_base64"] is not None


async def test_fetch_deduplicates_stories(monkeypatch: pytest.MonkeyPatch) -> None:
    import feedparser
    from worker.workflows.finance_newsletter import fetch_financial_news

    entries = [
        {
            "title": "Fed holds rates steady",
            "link": "https://example.com/fed",
            "summary": "The Federal Reserve kept rates unchanged.",
        },
        {
            "title": "Fed holds rates steady",
            "link": "https://example.com/fed-dup",
            "summary": "Same headline, different URL.",
        },
    ]

    class FakeFeed:
        pass

    feed = FakeFeed()
    feed.entries = entries
    monkeypatch.setattr(feedparser, "parse", lambda _url: feed)

    stories = await fetch_financial_news()

    assert len(stories) == 1
    assert stories[0].url == "https://example.com/fed"


def test_newsletter_prompt_lists_all_stories() -> None:
    from worker.workflows.finance_newsletter import newsletter_prompt

    prompt = newsletter_prompt(STORIES)
    assert "Fed holds rates steady" in prompt
    assert "https://example.com/markets" in prompt
    assert "podcast_script" in prompt
