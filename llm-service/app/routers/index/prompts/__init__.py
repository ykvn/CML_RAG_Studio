#
#  CLOUDERA APPLIED MACHINE LEARNING PROTOTYPE (AMP)
#  (C) Cloudera, Inc. 2026
#  All rights reserved.
#
#  Applicable Open Source License: Apache 2.0
#
#  REST endpoints for viewing and editing the LLM prompts used by the
#  application (backing the "Model Prompt" settings page).
#
import logging
from typing import Annotated, Any, Dict, List, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from .... import exceptions
from ....services.query import prompt_registry
from ....services.utils import has_admin_rights

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/prompts", tags=["Prompts"])


class Prompt(BaseModel):
    """A single editable prompt and its current value."""

    key: str
    title: str
    description: str
    variables: List[str]
    default_text: str
    text: str
    is_modified: bool


class PromptUpdates(BaseModel):
    """Mapping of prompt key -> new prompt text."""

    prompts: Dict[str, str]


class PromptReset(BaseModel):
    """Optional list of prompt keys to reset; empty/omitted resets all."""

    keys: Optional[List[str]] = None


@router.get(
    "",
    summary="Returns all editable prompts with their current values.",
    response_model=None,
)
@exceptions.propagates
def get_prompts() -> List[Prompt]:
    return [Prompt(**p) for p in prompt_registry.get_all_prompts()]


@router.put(
    "",
    summary="Saves updated prompt texts.",
    response_model=None,
)
@exceptions.propagates
def update_prompts(
    updates: PromptUpdates,
    remote_user: Annotated[str | None, Header()] = None,
    remote_user_perm: Annotated[str, Header()] = None,
) -> List[Prompt]:
    if not has_admin_rights(remote_user, remote_user_perm):
        raise HTTPException(
            status_code=401, detail="You do not have permission to edit prompts."
        )
    try:
        prompt_registry.save_prompts(updates.prompts)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    logger.info(
        "Prompts updated by %s: %s", remote_user, ", ".join(sorted(updates.prompts))
    )
    return [Prompt(**p) for p in prompt_registry.get_all_prompts()]


@router.post(
    "/reset",
    summary="Resets prompts back to their default texts.",
    response_model=None,
)
@exceptions.propagates
def reset_prompts(
    reset: Optional[PromptReset] = None,
    remote_user: Annotated[str | None, Header()] = None,
    remote_user_perm: Annotated[str, Header()] = None,
) -> List[Prompt]:
    if not has_admin_rights(remote_user, remote_user_perm):
        raise HTTPException(
            status_code=401, detail="You do not have permission to reset prompts."
        )
    keys = reset.keys if reset else None
    prompt_registry.reset_prompts(keys)
    logger.info(
        "Prompts reset by %s: %s",
        remote_user,
        "all" if keys is None else ", ".join(sorted(keys)),
    )
    return [Prompt(**p) for p in prompt_registry.get_all_prompts()]
