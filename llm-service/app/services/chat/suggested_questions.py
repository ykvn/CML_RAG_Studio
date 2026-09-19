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

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from random import shuffle
from typing import List, Optional

from app.ai.vector_stores.vector_store_factory import VectorStoreFactory
from app.services import llm_completion
from app.services.chat.utils import retrieve_chat_history, process_response
from app.services.metadata_apis import session_metadata_api
from app.services.metadata_apis.session_metadata_api import Session
from app.services.query import querier
from app.services.query.query_configuration import QueryConfiguration

logger = logging.getLogger(__name__)

SAMPLE_QUESTIONS = [
    "Apa aspirasi jangka panjang BNI di tahun 2028 dan pendekatan 4D yang digunakan dalam menyusun Corporate Plan 2024-2028?",
    "Apa saja 6 (enam) strategi utama BNI yang ditetapkan untuk tahun 2024-2028?",
    "Bagaimana pencapaian dan peran aplikasi 'wondr by BNI' serta platform 'BNIdirect' dalam transformasi digital BNI tahun 2024-2028?",
    "Bagaimana peran BNI Xpora dan jaringan kantor Luar Negeri (KLN) BNI dalam mendukung UKM Indonesia menembus pasar global?",
    "Apa fungsi dan cakupan siklus kredit yang dikelola melalui Loan Management System (LMS) Wholesale BNI?",
    "Bagaimana BNI menerapkan konsep hybrid branch dan inovasi e-channel untuk mengoptimalkan operasional outlet-nya?",
    "Bagaimana BNI memperkuat human capital dan produktivitas Relationship Manager (RM) melalui penggunaan tools seperti Connect dan Digisales?",
]


def generate_dummy_suggested_questions() -> List[str]:
    questions = SAMPLE_QUESTIONS.copy()
    shuffle(questions)
    return questions[:4]


def _generate_suggested_questions_direct_llm(session: Session) -> List[str]:
    chat_history = retrieve_chat_history(session.id)
    if not chat_history:
        return generate_dummy_suggested_questions()
    from .query.prompt_registry import SUGGESTED_QUESTIONS_DIRECT_PROMPT, get_prompt

    query_str = get_prompt(SUGGESTED_QUESTIONS_DIRECT_PROMPT)
    chat_response = llm_completion.completion(
        session.id, query_str, session.inference_model
    )
    suggested_questions = process_response(chat_response.message.content)
    return suggested_questions


def _session_has_content(session: Session, max_workers: int = 4) -> bool:
    """Return True if any of the session's data sources has indexed content.

    The content probe issues a Qdrant round-trip per data source, so check
    sources concurrently and stop as soon as one reports content.  This avoids a
    long serial chain of HTTP calls purely to answer "is there anything?".
    """
    data_source_ids = list(session.get_all_data_source_ids())
    if not data_source_ids:
        return False

    def _has_any_content(ds_id: int) -> bool:
        try:
            return (VectorStoreFactory.for_chunks(ds_id).size() or 0) > 0
        except Exception as exc:  # A failing probe must not block suggestions.
            logger.warning(
                "Failed to probe data source %s for content while generating "
                "suggested questions: %s",
                ds_id,
                exc,
            )
            return False

    if len(data_source_ids) == 1:
        return _has_any_content(data_source_ids[0])

    with ThreadPoolExecutor(max_workers=min(max_workers, len(data_source_ids))) as pool:
        futures = {
            pool.submit(_has_any_content, ds_id): ds_id for ds_id in data_source_ids
        }
        for future in as_completed(futures):
            if future.result():
                return True
    return False


def generate_suggested_questions(
    session_id: Optional[int],
    user_name: Optional[str] = None,
) -> List[str]:
    if session_id is None:
        return generate_dummy_suggested_questions()
    session = session_metadata_api.get_session(session_id, user_name)
    if len(session.get_all_data_source_ids()) == 0:
        return _generate_suggested_questions_direct_llm(session)

    if not _session_has_content(session):
        return _generate_suggested_questions_direct_llm(session)
        # raise HTTPException(status_code=404, detail="Knowledge base not found.")

    chat_history = retrieve_chat_history(session_id)
    query_str = (
        "Berikan daftar pertanyaan lanjutan yang mungkin relevan."
        " Setiap pertanyaan harus ditulis pada baris baru."
        " Tidak boleh ada lebih dari empat (4) pertanyaan."
        " Setiap pertanyaan tidak boleh lebih dari lima belas (15) kata."
        " Respons harus berupa daftar bullet, dengan tanda bintang (*) sebagai bullet."
        " Jangan memulai respons seperti ini - Berikut adalah empat pertanyaan yang dapat saya jawab berdasarkan informasi yang tersedia"
        " Hanya kembalikan daftar pertanyaan."
        " Hanya gunakan plain text."
        " Do not return any HTML tags or markdown formatting."
        " Do not return questions based on the metadata of the document. Only the content."
        " Do not start like this - `Here are four questions that I can answer based on the context information`"
        " Only return the list."
    )
    if chat_history:
        query_str = (
            query_str
            + (
                "I will provide a response from my last question to help with generating new questions."
                " Consider returning questions that are relevant to the response"
                " They might be follow up questions or questions that are related to the response."
                " Here is the last response received:\n"
            )
            + chat_history[-1].content
        )
    response, _ = querier.query(
        session,
        query_str,
        QueryConfiguration(
            top_k=session.response_chunks,
            model_name=session.inference_model,
            rerank_model_name=None,
            exclude_knowledge_base=False,
            use_question_condensing=False,
            use_hyde=False,
            use_postprocessor=False,
            use_tool_calling=False,
        ),
        [],
        should_condense_question=False,
    )
    suggested_questions = process_response(response.response)
    return suggested_questions
