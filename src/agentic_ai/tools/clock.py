"""Current date/time — LLMs don't know what day it is."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

from .base import Tool


class NoArgs(BaseModel):
    pass


def _now() -> str:
    return datetime.now().astimezone().strftime("%A, %Y-%m-%d %H:%M %Z")


datetime_tool = Tool("current_datetime", "Get the current local date and time.", NoArgs, _now)
