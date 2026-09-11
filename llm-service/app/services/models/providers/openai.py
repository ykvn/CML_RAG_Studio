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
import os
from typing import Optional

import httpx
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai_like import OpenAILike

from ._model_provider import _ModelProvider
from ...caii.types import ModelResponse
from ...llama_utils import completion_to_prompt, messages_to_prompt
from ....config import settings, ModelSource


class OpenAiModelProvider(_ModelProvider):
    @staticmethod
    def get_env_var_names() -> set[str]:
        return {"OPENAI_API_KEY"}

    @staticmethod
    def get_model_source() -> ModelSource:
        return ModelSource.OPENAI

    @staticmethod
    def get_priority() -> int:
        return 2

    @staticmethod
    def list_llm_models() -> list[ModelResponse]:
        return [
            ModelResponse(
                model_id="Qwen3.8-27B",
                name="Qwen3.8-27B",
                tool_calling_supported=True,
            ),
            ModelResponse(
                model_id="Qwen3.8-27B-no-thinking",
                name="Qwen3.8-27B-no-thinking",
                tool_calling_supported=True,
            ),
            ModelResponse(
                model_id="Qwen3.8-27B-low",
                name="Qwen3.8-27B-low",
                tool_calling_supported=True,
            ),
            ModelResponse(
                model_id="Qwen3.8-27B-medium",
                name="Qwen3.8-27B-medium",
                tool_calling_supported=True,
            ),
            ModelResponse(
                model_id="Qwen3.8-27B-xhigh",
                name="Qwen3.8-27B-xhigh",
                tool_calling_supported=True,
            ),
        ]

    @staticmethod
    def list_embedding_models() -> list[ModelResponse]:
        return [
            ModelResponse(
                model_id="bge-m3",
                name="bge-m3",
            ),
        ]

    @staticmethod
    def list_reranking_models() -> list[ModelResponse]:
        return []

    @staticmethod
    def _http_client() -> Optional[httpx.Client]:
        timeout = httpx.Timeout(120.0, connect=10.0)
        limits = httpx.Limits(max_keepalive_connections=10, max_connections=20)
        
        if os.path.exists("/etc/ssl/certs/ca-certificates.crt"):
            return httpx.Client(
                verify="/etc/ssl/certs/ca-certificates.crt",
                timeout=timeout,
                limits=limits,
            )
        else:
            return httpx.Client(timeout=timeout, limits=limits)

    @staticmethod
    def get_llm_model(name: str) -> OpenAILike:
        return OpenAILike(
            model=name,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            max_tokens=2048,
            context_window=32768,
            is_chat_model=True,
            is_function_calling_model=True,
            api_base=settings.openai_api_base,
            api_key=settings.openai_api_key,
            timeout=settings.llm_request_timeout,
            max_retries=settings.llm_max_retries,
            http_client=OpenAiModelProvider._http_client(),
        )

    @staticmethod
    def get_embedding_model(name: str) -> OpenAIEmbedding:
        return OpenAIEmbedding(
            model_name=name,
            api_key=settings.openai_api_key,
            api_base=settings.openai_api_base,
            timeout=settings.llm_request_timeout,
            max_retries=settings.llm_max_retries,
            embed_batch_size=settings.embedding_batch_size,
            http_client=OpenAiModelProvider._http_client(),
        )

    @staticmethod
    def get_reranking_model(name: str, top_n: int) -> BaseNodePostprocessor:
        raise NotImplementedError("No reranking models available")


# ensure interface is implemented
_ = OpenAiModelProvider()
