#
#  CLOUDERA APPLIED MACHINE LEARNING PROTOTYPE (AMP)
#  (C) Cloudera, Inc. 2024
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
import logging
from typing import Any, Optional, List, Tuple

from llama_index.core import PromptTemplate
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    MessageRole,
    ChatResponseGen,
)
from llama_index.core.base.response.schema import Response
from llama_index.core.chat_engine import (
    CondensePlusContextChatEngine,
)
from llama_index.core.chat_engine.types import (
    StreamingAgentChatResponse,
)
from llama_index.core.llms import LLM
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.tools import ToolOutput

from .query_configuration import QueryConfiguration
from .simple_reranker import SimpleReranker
from .. import llm_completion, models

logger = logging.getLogger(__name__)

CUSTOM_CONTEXT_PROMPT_TEMPLATE = """\
Berikut adalah percakapan antara pengguna dan asisten AI. \
Asisten memberikan jawaban secara detail dan spesifik berdasarkan konteks yang diberikan. \
Jika asisten tidak mengetahui jawaban dari suatu pertanyaan, asisten akan menyatakan bahwa ia tidak mengetahuinya.

Sebagai asisten, berikan jawaban hanya berdasarkan sumber-sumber yang diberikan dengan \
menyertakan sitasi pada paragraf. Saat mereferensikan informasi dari sebuah sumber, \
sebutkan sumber yang sesuai menggunakan ID masing-masing. \
Setiap jawaban atau paragraf harus menyertakan setidaknya satu sitasi sumber. \
Hanya buat sitasi jika Anda secara eksplisit mereferensikannya. \
Sitasi harus menggunakan tag anchor (<a class="rag_citation" href="CITATION_HERE"></a>) \
dan (SANGAT PENTING) diletakkan langsung di dalam teks (in-line). Jangan gunakan catatan kaki atau catatan akhir. \
Jika tidak ada sumber yang membantu, nyatakan hal tersebut. \
Jangan membuat ID sumber buatan. Hanya gunakan ID sumber yang tersedia pada konteks.

Aturan Tambahan Jawaban:
Di bagian paling akhir setiap jawaban, WAJIB buat tag XML <followups> yang berisi 4–5 opsi pertanyaan lanjutan yang interaktif, kontekstual, dan spesifik terkait data atau topik yang baru saja dijelaskan. Pisahkan setiap pertanyaan dengan karakter pipe (|). JANGAN gunakan list markdown di dalam tag ini.

Format keluaran akhir:
[Jawaban Anda di sini...]
<followups>Pertanyaan drill-down spesifik?|Pertanyaan perbandingan produk/kategori?|Pertanyaan tren waktu?|Pertanyaan dampak/penyebab spesifik?</followups>

Sebagai contoh:

<Contexts>
Source: 1
Langit berwarna merah di sore hari dan biru di pagi hari.

Source: 2
Air terasa basah ketika langit berwarna merah.

<Query>
Kapan air terasa basah?

<Answer>
Air akan terasa basah ketika langit berwarna merah<a class="rag_citation" href="1"></a>, \
yang terjadi pada sore hari<a class="rag_citation" href="2"></a>.
<followups>Berapa suhu air saat langit berwarna merah?|Apakah air tetap basah pada pagi hari?|Mengapa warna langit berubah menjadi biru di pagi hari?|Apa yang menyebabkan langit berwarna merah di sore hari?</followups>

Sekarang giliran Anda. Di bawah ini adalah beberapa sumber informasi terhitung:

<Contexts>
{context_str}

<Query>
{query_str}

<Answer>
"""


CUSTOM_CONTEXT_REFINE_PROMPT_TEMPLATE = """\
Berikut adalah percakapan antara pengguna dan asisten AI. \
Asisten memberikan jawaban secara detail dan spesifik berdasarkan konteks yang diberikan. \
Jika asisten tidak mengetahui jawaban dari suatu pertanyaan, asisten akan menyatakan bahwa ia tidak mengetahuinya.

Sebagai asisten, berikan jawaban hanya berdasarkan sumber-sumber yang diberikan dengan \
menyertakan sitasi pada paragraf. Saat mereferensikan informasi dari sebuah sumber, \
sebutkan sumber yang sesuai menggunakan ID masing-masing. \
Setiap jawaban atau paragraf harus menyertakan setidaknya satu sitasi sumber. \
Hanya buat sitasi jika Anda secara eksplisit mereferensikannya. \
Sitasi harus menggunakan tag anchor (<a class="rag_citation" href="CITATION_HERE"></a>) \
dan (SANGAT PENTING) diletakkan langsung di dalam teks (in-line). Jangan gunakan catatan kaki atau catatan akhir. \
Jika tidak ada sumber yang membantu, nyatakan hal tersebut. \
Jangan membuat ID sumber buatan. Hanya gunakan ID sumber yang tersedia pada konteks.

Aturan Tambahan Jawaban:
Di bagian paling akhir setiap jawaban, WAJIB buat tag XML <followups> yang berisi 4–5 opsi pertanyaan lanjutan yang interaktif, kontekstual, dan spesifik terkait data atau topik yang baru saja dijelaskan. Pisahkan setiap pertanyaan dengan karakter pipe (|). JANGAN gunakan list markdown di dalam tag ini.

Format keluaran akhir:
[Jawaban Anda di sini...]
<followups>Pertanyaan drill-down spesifik?|Pertanyaan perbandingan produk/kategori?|Pertanyaan tren waktu?|Pertanyaan dampak/penyebab spesifik?</followups>

Sekarang giliran Anda. Kami telah menyediakan jawaban yang sudah ada sebelumnya:

<Existing Answer>
{existing_answer}

Di bawah ini adalah beberapa sumber informasi terhitung.
Gunakan sumber tersebut untuk memperjelas jawaban yang ada.
Jika sumber yang diberikan tidak membantu, ulangi kembali jawaban yang sudah ada.
Mulai perjelas!

<Contexts>
{context_msg}

<Query>
{query_str}

<Answer>
"""

CUSTOM_CONDENSE_TEMPLATE = """\
Berdasarkan percakapan (antara Pengguna dan Asisten) serta pesan lanjutan dari Pengguna, \
tulis ulang pesan tersebut menjadi pertanyaan mandiri yang mencakup seluruh konteks \
relevan dari percakapan. Berikan pertanyaannya saja, tanpa penjelasan atau deskripsi tambahan.

<Riwayat Obrolan>
{chat_history}

<Pesan Lanjutan>
{question}

<Pertanyaan Mandiri>
"""

CUSTOM_CONDENSE_PROMPT = PromptTemplate(CUSTOM_CONDENSE_TEMPLATE)
CUSTOM_CONTEXT_PROMPT = PromptTemplate(CUSTOM_CONTEXT_PROMPT_TEMPLATE)
CUSTOM_CONTEXT_REFINE_PROMPT = PromptTemplate(CUSTOM_CONTEXT_REFINE_PROMPT_TEMPLATE)


def _resolve_prompts() -> Tuple[PromptTemplate, PromptTemplate, PromptTemplate]:
    """Resolve chat prompts from the prompt registry, honouring user overrides."""
    from .prompt_registry import (
        RAG_CONDENSE_PROMPT as CONDENSE_KEY,
        RAG_CONTEXT_PROMPT as CONTEXT_KEY,
        RAG_CONTEXT_REFINE_PROMPT as REFINE_KEY,
        get_prompt,
    )

    return (
        PromptTemplate(get_prompt(CONTEXT_KEY)),
        PromptTemplate(get_prompt(REFINE_KEY)),
        PromptTemplate(get_prompt(CONDENSE_KEY)),
    )


class FlexibleContextChatEngine(CondensePlusContextChatEngine):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._configuration: QueryConfiguration = QueryConfiguration()

    def stream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> StreamingAgentChatResponse:
        if self._configuration.use_streaming:
            streaming_response: StreamingAgentChatResponse = super().stream_chat(message, chat_history)
            return streaming_response

        synthesizer, context_source, context_nodes = self._run_c3(message, chat_history)

        response: Response = synthesizer.synthesize(message, context_nodes)

        def wrapped_gen(res: Response) -> ChatResponseGen:
            assistant_message = ChatMessage(
                content=res.response, role=MessageRole.ASSISTANT
            )
            yield ChatResponse(
                message=assistant_message,
                delta="",
            )

            user_message = ChatMessage(content=message, role=MessageRole.USER)
            self._memory.put(user_message)
            self._memory.put(assistant_message)

        return StreamingAgentChatResponse(
            response=str(response),
            chat_stream=wrapped_gen(response),
            sources=[context_source],
            source_nodes=context_nodes,
        )

    def condense_question(
        self, chat_history: List[ChatMessage], latest_message: str
    ) -> str:
        return super()._condense_question(chat_history, latest_message)

    def _run_c3(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        streaming: bool = False,
    ) -> Tuple[CompactAndRefine, ToolOutput, List[NodeWithScore]]:
        if chat_history is not None:
            self._memory.set(chat_history)

        chat_history = self._memory.get(input=message)

        # Condense conversation history and latest message to a standalone question
        vector_match_input = message
        if self._configuration.use_question_condensing:
            vector_match_input = self._condense_question(chat_history, message)
            if self._verbose:
                logger.info(f"Condensed question: {vector_match_input}")

        # get the context nodes using the condensed question
        if self._configuration.use_hyde:
            vector_match_input = llm_completion.hypothetical(
                vector_match_input, self._configuration
            )
            if self._verbose:
                logger.info(f"Hypothetical document: {vector_match_input}")

        context_nodes = self._get_nodes(vector_match_input)
        for node in context_nodes:
            # number the nodes in the content
            new_content = f"Source: {node.node.node_id}\n{node.node.get_content()}\n"
            node.node.set_content(value=new_content)
        context_source = ToolOutput(
            tool_name="retriever",
            content=str(context_nodes),
            raw_input={"message": vector_match_input},
            raw_output=context_nodes,
        )

        # build the response synthesizer
        response_synthesizer = self._get_response_synthesizer(
            chat_history, streaming=streaming
        )

        return response_synthesizer, context_source, context_nodes

    @property
    def retriever(self) -> BaseRetriever:
        return self._retriever

    @property
    def node_postprocessors(self) -> List[BaseNodePostprocessor]:
        return self._node_postprocessors


def build_flexible_chat_engine(
    configuration: QueryConfiguration,
    llm: LLM,
    retriever: Optional[BaseRetriever],
) -> Optional[FlexibleContextChatEngine]:
    if not retriever:
        return None
    postprocessors = _create_node_postprocessors(configuration)
    context_prompt, context_refine_prompt, condense_prompt = _resolve_prompts()
    chat_engine: FlexibleContextChatEngine = FlexibleContextChatEngine.from_defaults(
        llm=llm,
        context_prompt=context_prompt,
        context_refine_prompt=context_refine_prompt,
        condense_prompt=condense_prompt,
        retriever=retriever,
        node_postprocessors=postprocessors,
    )
    chat_engine._configuration = configuration
    return chat_engine


class DebugNodePostProcessor(BaseNodePostprocessor):
    def _postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle] = None
    ) -> list[NodeWithScore]:
        logger.debug(f"nodes: {len(nodes)}")
        for node in sorted(nodes, key=lambda n: n.node.node_id):
            logger.debug(
                node.node.node_id, node.node.metadata["document_id"], node.score
            )

        return nodes


def _create_node_postprocessors(
    configuration: QueryConfiguration,
) -> list[BaseNodePostprocessor]:
    if not configuration.use_postprocessor:
        return []

    if configuration.rerank_model_name is None:
        return [DebugNodePostProcessor(), SimpleReranker(top_n=configuration.top_k)]

    return [
        DebugNodePostProcessor(),
        models.Reranking.get(
            model_name=configuration.rerank_model_name,
            top_n=configuration.top_k,
        )
        or SimpleReranker(top_n=configuration.top_k),
        DebugNodePostProcessor(),
    ]
