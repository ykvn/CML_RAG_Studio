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
#  Cloudera.  Used apart from the collective work, this file is
#  licensed for your use pursuant to the open source license
#  identified above.
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
import pytest
import qdrant_client as q_client

from app.ai.vector_stores.qdrant import _new_qdrant_client


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure external qdrant env vars are unset by default for these tests."""
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)


class TestQdrantClient:
    def test_external_url_constructs_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VECTOR_DB_PROVIDER", "EXTERNAL_QDRANT")
        monkeypatch.setenv("QDRANT_URL", "https://qdrant.example.com")
        monkeypatch.setenv("QDRANT_API_KEY", "secret-key")
        monkeypatch.setenv("CDSW_APIV2_KEY", "cml-token")

        client = _new_qdrant_client()

        assert isinstance(client, q_client.QdrantClient)
        # HTTPS endpoint: must connect on port 443, not the library default 6333
        assert client._client._port == 443
        # The API key must be sent as the api-key header.
        assert client._client._api_key == "secret-key"
        # SSL certificate verification must be disabled for the REST client.
        assert client._client._rest_args.get("verify") is False
        # CML SSO gateway: the Bearer token must be passed as Authorization.
        assert client._client._rest_args.get("headers") == {
            "Authorization": "Bearer cml-token"
        }

    def test_external_provider_requires_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VECTOR_DB_PROVIDER", "EXTERNAL_QDRANT")
        monkeypatch.delenv("QDRANT_URL", raising=False)

        with pytest.raises(ValueError, match="QDRANT_URL"):
            _new_qdrant_client()