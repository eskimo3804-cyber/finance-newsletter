"""Weekly finance newsletter: top financial news by email with an audio recap attachment."""

import base64
from datetime import timedelta

import pydantic
from mistralai import workflows
from mistralai.workflows import Depends
from mistralai.workflows.plugins.mistralai import (
    ChatCompletionRequest,
    SystemMessage,
    UserMessage,
    chat_parse_to_model,
)
from mistralai.workflows.plugins.mistralai.connectors import (
    ToolCallClient,
    connector,
    uses_connectors,
)

gmail = connector("gmail", auto_auth=False)

RSS_FEEDS: tuple[str, ...] = (
    "https://finance.yahoo.com/news/rssindex",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "https://www.marketwatch.com/rss/topstories",
)

PODCAST_VOICE = "en-US-AndrewMultilingualNeural"
MAX_ATTACHMENT_BYTES = 2_200_000


class Story(pydantic.BaseModel):
    title: str
    url: str
    summary: str


class NewsletterContent(pydantic.BaseModel):
    subject: str
    intro: str
    top_stories: list[Story]
    podcast_script: str
    plain_text: str


class NewsletterInput(pydantic.BaseModel):
    recipient: str = "amaury.kezachian@outlook.com"
    with_audio: bool = True
    dry_run: bool = True


class NewsletterOutput(pydantic.BaseModel):
    subject: str
    newsletter: NewsletterContent
    audio_generated: bool
    email_sent: bool
    recipient: str | None = None


@workflows.activity(
    name="fetch-financial-news", start_to_close_timeout=timedelta(seconds=90)
)
async def fetch_financial_news() -> list[Story]:
    import feedparser

    stories: dict[str, Story] = {}
    for feed_url in RSS_FEEDS:
        parsed = feedparser.parse(feed_url)
        for entry in parsed.entries:
            title = (entry.get("title") or "").strip()
            link = (entry.get("link") or "").strip()
            summary = (entry.get("summary") or "").strip()[:600]
            if not title or not link or title.lower() in stories:
                continue
            stories[title.lower()] = Story(title=title, url=link, summary=summary)
    return list(stories.values())[:40]


def newsletter_prompt(stories: list[Story]) -> str:
    items = "\n".join(
        f"- {story.title}\n  Link: {story.url}\n  Summary: {story.summary}"
        for story in stories
    )
    return (
        "From the financial news items below, write this week's finance newsletter.\n"
        "Rules:\n"
        "- Pick the 5 most important stories of the week (markets, macro, companies, policy).\n"
        "- Each summary is 2-3 sentences, factual and neutral, mentioning key numbers.\n"
        "- podcast_script: a 160-200 word spoken audio recap covering the same top 5 stories. "
        "Conversational podcast tone, no stage directions, no bullet points, no URLs, plain "
        "text only suitable for text-to-speech.\n"
        "- subject: a short catchy newsletter subject line.\n"
        "- intro: one short paragraph greeting the reader.\n"
        "- plain_text: the full newsletter as plain text.\n"
        "News items:\n" + items
    )


@workflows.activity(
    name="generate-audio-recap", start_to_close_timeout=timedelta(seconds=300)
)
async def generate_audio(script: str) -> str:
    import edge_tts

    communicate = edge_tts.Communicate(script, PODCAST_VOICE)
    chunks: list[bytes] = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            chunks.append(chunk["data"])
    audio = b"".join(chunks)
    if not audio:
        raise RuntimeError("text-to-speech produced no audio")
    return base64.b64encode(audio).decode("ascii")


@workflows.activity(
    name="send-newsletter-email", start_to_close_timeout=timedelta(seconds=120)
)
async def send_newsletter_email(
    recipient: str,
    subject: str,
    html: str,
    plain_text: str,
    audio_base64: str | None,
    client: ToolCallClient = Depends(gmail),  # noqa: B008
) -> None:
    arguments: dict = {
        "to": [recipient],
        "subject": subject,
        "body": plain_text,
        "htmlBody": html,
    }
    if audio_base64 is not None:
        arguments["attachments"] = [
            {
                "filename": "weekly-finance-recap.mp3",
                "mimeType": "audio/mpeg",
                "content": audio_base64,
                "contentEncoding": "base64",
            }
        ]
    await client.call_tool(tool_name="send_email", arguments=arguments)


def render_html(content: NewsletterContent) -> str:
    stories = "\n".join(
        f"""<li style="margin-bottom:18px">
          <a href="{story.url}" style="color:#0a3d62;font-weight:600;text-decoration:none">
            {pydantic_escape(story.title)}
          </a>
          <p style="margin:6px 0 0;color:#333">{pydantic_escape(story.summary)}</p>
        </li>"""
        for story in content.top_stories
    )
    return f"""\
<div style="font-family:Helvetica,Arial,sans-serif;max-width:640px;margin:0 auto">
  <h1 style="color:#0a3d62;border-bottom:3px solid #f2a154;padding-bottom:8px">
    Weekly Finance Newsletter
  </h1>
  <p style="color:#333">{pydantic_escape(content.intro)}</p>
  <h2 style="color:#0a3d62">Top stories of the week</h2>
  <ul style="list-style:none;padding:0">{stories}</ul>
  <p style="color:#666;margin-top:24px">
    &#127911; Your audio recap is attached to this email: weekly-finance-recap.mp3
  </p>
</div>"""


def pydantic_escape(text: str) -> str:
    import html

    return html.escape(text)


@workflows.workflow.define(name="weekly-finance-newsletter")
@uses_connectors(gmail)
class WeeklyFinanceNewsletterWorkflow:
    @workflows.workflow.entrypoint
    async def run(self, data: NewsletterInput) -> NewsletterOutput:
        stories = await fetch_financial_news()
        if not stories:
            raise RuntimeError("no financial news could be fetched")

        request = ChatCompletionRequest(
            model="mistral-medium-latest",
            messages=[
                SystemMessage(
                    content=(
                        "You are a financial newsletter editor. Always answer with a "
                        "single JSON object matching the requested schema."
                    )
                ),
                UserMessage(content=newsletter_prompt(stories)),
            ],
        )
        content = await chat_parse_to_model(NewsletterContent, request)

        audio_generated = False
        audio_base64: str | None = None
        if data.with_audio and len(content.podcast_script) > 0:
            candidate = await generate_audio(content.podcast_script)
            if len(candidate) <= (4 * ((MAX_ATTACHMENT_BYTES + 2) // 3)):
                audio_base64 = candidate
                audio_generated = True

        html = render_html(content)
        plain = content.plain_text or content.subject

        if data.dry_run:
            return NewsletterOutput(
                subject=content.subject,
                newsletter=content,
                audio_generated=audio_generated,
                email_sent=False,
                recipient=None,
            )

        await send_newsletter_email(
            recipient=data.recipient,
            subject=content.subject,
            html=html,
            plain_text=plain,
            audio_base64=audio_base64,
        )
        return NewsletterOutput(
            subject=content.subject,
            newsletter=content,
            audio_generated=audio_generated,
            email_sent=True,
            recipient=data.recipient,
        )
