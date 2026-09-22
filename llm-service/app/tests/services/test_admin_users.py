#
#  CLOUDERA APPLIED MACHINE LEARNING PROTOTYPE (AMP)
#  (C) Cloudera, Inc. 2025
#  All rights reserved.
#
#  Applicable Open Source License: Apache 2.0
#
#  NOTE: Cloudera open source products are modular software products
#  made up of hundreds of individual components, each of which was
#  individually copyrighted.  Each Cloudera open source product is a
#  collective work under U.S. Copyright Law. Your license to use the
#  collective work is as provided in your written agreement with
#  Cloudera.
#
#  This code is provided to you pursuant a written agreement with
#  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute
#  this code. If you do not have a written agreement with Cloudera nor
#  with an authorized and properly licensed third party, you do not
#  have any rights to access nor to use this code.
#
#  Absent a written agreement with Cloudera, Inc. ("Cloudera") to the
#  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
#  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED
#  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO
#  IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND
#  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU,
#  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS
#  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE
#  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
#  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES
#  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF
#  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
#  DATA.
#

"""Tests for app/services/admin_users.py (cosmetic UI admin list)."""

import json

import pytest

from app.services import admin_users
from app.services.admin_users import is_admin, load_admin_users


@pytest.fixture(autouse=True)
def clear_caches() -> None:
    admin_users._cached_env_admins.cache_clear()
    admin_users._cached_file_admins.cache_clear()


class TestLoadAdminUsers:
    def test_empty_env_allows_everyone(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        monkeypatch.delenv("RAG_ADMINS", raising=False)
        monkeypatch.setenv("RAG_ADMINS_FILE", str(tmp_path / "does_not_exist.json"))
        assert is_admin("anyone@example.com") is True

    def test_env_var_admins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RAG_ADMINS", "Alice@Example.com, bob@example.com ,,")
        assert load_admin_users() == frozenset({"alice@example.com", "bob@example.com"})

    def test_env_var_takes_precedence_over_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        admin_file = tmp_path / "admins.json"
        admin_file.write_text(json.dumps({"admins": ["carol@example.com"]}))
        monkeypatch.setenv("RAG_ADMINS", "alice@example.com")
        monkeypatch.setenv("RAG_ADMINS_FILE", str(admin_file))

        assert load_admin_users() == frozenset({"alice@example.com"})

    def test_reads_admins_file(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        admin_file = tmp_path / "admins.json"
        admin_file.write_text(
            json.dumps({"admins": ["Alice@Example.com", "bob@example.com"]})
        )
        monkeypatch.delenv("RAG_ADMINS", raising=False)
        monkeypatch.setenv("RAG_ADMINS_FILE", str(admin_file))

        assert load_admin_users() == frozenset(
            {"alice@example.com", "bob@example.com"}
        )

    def test_missing_file_allows_everyone(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        monkeypatch.delenv("RAG_ADMINS", raising=False)
        monkeypatch.setenv("RAG_ADMINS_FILE", str(tmp_path / "does_not_exist.json"))
        assert load_admin_users() == frozenset()
        assert is_admin("anyone@example.com") is True

    def test_empty_admins_list_allows_everyone(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        admin_file = tmp_path / "admins.json"
        admin_file.write_text(json.dumps({"admins": []}))
        monkeypatch.delenv("RAG_ADMINS", raising=False)
        monkeypatch.setenv("RAG_ADMINS_FILE", str(admin_file))
        assert is_admin("anyone@example.com") is True

    def test_invalid_json_allows_everyone(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        admin_file = tmp_path / "admins.json"
        admin_file.write_text("{ not json")
        monkeypatch.delenv("RAG_ADMINS", raising=False)
        monkeypatch.setenv("RAG_ADMINS_FILE", str(admin_file))
        assert is_admin("anyone@example.com") is True


class TestIsAdmin:
    @pytest.fixture()
    def admin_file(self, monkeypatch: pytest.MonkeyPatch, tmp_path):
        admin_file = tmp_path / "admins.json"
        admin_file.write_text(json.dumps({"admins": ["alice@example.com"]}))
        monkeypatch.delenv("RAG_ADMINS", raising=False)
        monkeypatch.setenv("RAG_ADMINS_FILE", str(admin_file))
        return admin_file

    def test_admin_user(self, admin_file) -> None:
        assert is_admin("Alice@Example.com") is True
        assert is_admin("  alice@example.com ") is True

    def test_non_admin_user(self, admin_file) -> None:
        assert is_admin("mallory@example.com") is False

    def test_none_username(self, admin_file) -> None:
        assert is_admin(None) is False

    def test_rereads_file_after_change(self, admin_file) -> None:
        assert is_admin("bob@example.com") is False
        admin_file.write_text(json.dumps({"admins": ["bob@example.com"]}))
        assert is_admin("bob@example.com") is True
