# ##############################################################################
#  CLOUDERA APPLIED MACHINE LEARNING PROTOTYPE (AMP)
#  (C) Cloudera, Inc. 2025
#  All rights reserved.
#
#  Applicable Open Source License: Apache 2.0
#
#
#  This code is provided to you pursuant a written agreement with
#  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute
#  this code. If you do not have a written agreement with Cloudera nor
#  with an authorized and properly licensed third party, you do not
#  have any rights to access nor to use this code.
#
#  Absent a written agreement with Cloudera, Inc. ("Cloudera") to the
#  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
#  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS OR IMPLIED WARRANTIES
#  WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO IMPLIED WARRANTIES
#  OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND FITNESS FOR A PARTICULAR
#  PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU, AND WILL NOT DEFEND, INDEMNIFY,
#  NOR HOLD YOU HARMLESS FOR ANY CLAIMS ARISING FROM OR RELATED TO THE CODE;
#  AND (D)WITH RESPECT TO YOUR EXERCISE OF ANY RIGHTS GRANTED TO YOU FOR THE
#  CODE, CLOUDERA IS NOT LIABLE FOR ANY LOSS OF PROFITS, LOSS OF INCOME, LOSS
#  OF BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
#  DATA.
# ##############################################################################
"""Admin user list used purely for UI gating of the Settings page.

This module is cosmetic only: it powers the read-only ``GET /amp/is-admin``
endpoint which the frontend uses to decide whether to show the Settings page.
It does NOT enforce any backend permissions; all existing permission checks
(e.g. ``has_admin_rights``) are unaffected.

The admin list comes from (first match wins):
1. The ``RAG_ADMINS`` environment variable (comma-separated usernames).
2. The ``admins.json`` file (path configurable via ``RAG_ADMINS_FILE``,
   defaulting to ``admins.json`` next to the ``llm-service`` directory).

If the list is empty or unavailable, *every* user is reported as an admin
(fail-open), preserving the pre-existing behavior for deployments that have
not opted in.
"""

import functools
import json
import logging
import os
from typing import FrozenSet, Optional

from app.config import settings

logger = logging.getLogger(__name__)

_CACHE_TTL_SECONDS = 30.0


def _normalize(username: Optional[str]) -> str:
    return (username or "").strip().lower()


def _parse_env_admins(raw: str) -> FrozenSet[str]:
    return frozenset(
        _normalize(admin) for admin in raw.split(",") if _normalize(admin)
    )


def _load_admins_from_file(path: str) -> FrozenSet[str]:
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        admins = data.get("admins", []) if isinstance(data, dict) else []
        if not isinstance(admins, list):
            logger.warning("Invalid 'admins' entry in %s; ignoring.", path)
            return frozenset()
        return frozenset(_normalize(admin) for admin in admins if _normalize(admin))
    except FileNotFoundError:
        return frozenset()
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Could not read admin list from %s: %s", path, e)
        return frozenset()


@functools.lru_cache(maxsize=1)
def _cached_env_admins(raw: str) -> FrozenSet[str]:
    return _parse_env_admins(raw)


@functools.lru_cache(maxsize=1)
def _cached_file_admins(path: str, mtime: float) -> FrozenSet[str]:
    return _load_admins_from_file(path)


def load_admin_users() -> FrozenSet[str]:
    """Return the configured admin usernames (normalized, lowercase).

    The result is cached briefly so the file can be edited without a restart.
    """
    env_raw = settings.rag_admins_env
    if env_raw:
        return _cached_env_admins(env_raw)

    path = settings.rag_admins_file
    try:
        mtime: Optional[float] = os.stat(path).st_mtime
    except OSError:
        mtime = None
    if mtime is None:
        _cached_file_admins.cache_clear()
        return frozenset()
    return _cached_file_admins(path, mtime)


def is_admin(username: Optional[str]) -> bool:
    """Whether the given user should see the Settings page in the UI.

    Fail-open: an empty or missing admin list means everyone is an admin.
    """
    admins = load_admin_users()
    if not admins:
        return True
    return _normalize(username) in admins
