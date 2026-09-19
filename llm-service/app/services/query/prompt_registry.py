#
#  CLOUDERA APPLIED MACHINE LEARNING PROTOTYPE (AMP)
#  (C) Cloudera, Inc. 2026
#  All rights reserved.
#
#  Applicable Open Source License: Apache 2.0
#
#  Central registry for all LLM prompts used by the application.
#  Each prompt can be overridden at runtime from the "Model Prompt" settings
#  page. Overrides are persisted in a prompts.json file and are picked up
#  on the next request (no application restart required).
#
import json
import os
from typing import Any, Dict, List, Optional

from ...config import settings

PROMPT_OVERRIDES_FILE = "prompts.json"

# Prompt ids
RAG_CONTEXT_PROMPT = "rag_context_prompt"
RAG_CONTEXT_REFINE_PROMPT = "rag_context_refine_prompt"
AGENT_SYSTEM_PROMPT = "agent_system_prompt"
DOCUMENT_SUMMARY_PROMPT = "document_summary_prompt"
OCR_PAGE_PROMPT = "ocr_page_prompt"
HYDE_PROMPT = "hyde_prompt"
SESSION_RENAME_PROMPT = "session_rename_prompt"
SUGGESTED_QUESTIONS_DIRECT_PROMPT = "suggested_questions_direct_prompt"


class PromptDefinition:
    def __init__(
        self,
        key: str,
        title: str,
        description: str,
        variables: List[str],
        default_text: str,
    ):
        self.key = key
        self.title = title
        self.description = description
        self.variables = variables
        self.default_text = default_text


PROMPT_DEFINITIONS: Dict[str, PromptDefinition] = {
    d.key: d
    for d in [
        PromptDefinition(
            key=RAG_CONTEXT_PROMPT,
            title="RAG Answer (Context)",
            description=(
                "Main system prompt used to answer user questions from retrieved "
                "knowledge-base chunks. Available variables: {context_str} "
                "(retrieved sources), {query_str} (the user's question)."
            ),
            variables=["context_str", "query_str"],
            default_text=(
                "Berikut adalah percakapan antara pengguna dan asisten AI. "
                "Asisten memberikan jawaban secara detail dan spesifik berdasarkan konteks yang diberikan. "
                "Jika asisten tidak mengetahui jawaban dari suatu pertanyaan, asisten akan menyatakan bahwa ia tidak mengetahuinya.\n\n"
                "Sebagai asisten, berikan jawaban hanya berdasarkan sumber-sumber yang diberikan dengan "
                "menyertakan sitasi pada paragraf. Saat mereferensikan informasi dari sebuah sumber, "
                "sebutkan sumber yang sesuai menggunakan ID masing-masing. "
                "Setiap jawaban atau paragraf harus menyertakan setidaknya satu sitasi sumber. "
                "Hanya buat sitasi jika Anda secara eksplisit mereferensikannya. "
                "Sitasi harus menggunakan tag anchor (<a class=\"rag_citation\" href=\"CITATION_HERE\"></a>) "
                "dan (SANGAT PENTING) diletakkan langsung di dalam teks (in-line). Jangan gunakan catatan kaki atau catatan akhir. "
                "Jika tidak ada sumber yang membantu, nyatakan hal tersebut. "
                "Jangan membuat ID sumber buatan. Hanya gunakan ID sumber yang tersedia pada konteks.\n\n"
                "Aturan Tambahan Jawaban:\n"
                "Di bagian paling akhir setiap jawaban, WAJIB buat tag XML <followups> yang berisi 4–5 opsi pertanyaan lanjutan yang interaktif, kontekstual, dan spesifik terkait data atau topik yang baru saja dijelaskan. Pisahkan setiap pertanyaan dengan karakter pipe (|). JANGAN gunakan list markdown di dalam tag ini.\n\n"
                "Format keluaran akhir:\n"
                "[Jawaban Anda di sini...]\n"
                "<followups>Pertanyaan drill-down spesifik?|Pertanyaan perbandingan produk/kategori?|Pertanyaan tren waktu?|Pertanyaan dampak/penyebab spesifik?</followups>\n\n"
                "Sebagai contoh:\n\n"
                "<Contexts>\n"
                "Source: 1\n"
                "Langit berwarna merah di sore hari dan biru di pagi hari.\n\n"
                "Source: 2\n"
                "Air terasa basah ketika langit berwarna merah.\n\n"
                "<Query>\n"
                "Kapan air terasa basah?\n\n"
                "<Answer>\n"
                "Air akan terasa basah ketika langit berwarna merah<a class=\"rag_citation\" href=\"1\"></a>, "
                "yang terjadi pada sore hari<a class=\"rag_citation\" href=\"2\"></a>.\n"
                "<followups>Berapa suhu air saat langit berwarna merah?|Apakah air tetap basah pada pagi hari?|Mengapa warna langit berubah menjadi biru di pagi hari?|Apa yang menyebabkan langit berwarna merah di sore hari?</followups>\n\n"
                "Sekarang giliran Anda. Di bawah ini adalah beberapa sumber informasi terhitung:\n\n"
                "<Contexts>\n"
                "{context_str}\n\n"
                "<Query>\n"
                "{query_str}\n\n"
                "<Answer>\n"
            ),
        ),
        PromptDefinition(
            key=RAG_CONTEXT_REFINE_PROMPT,
            title="RAG Answer Refine",
            description=(
                "Used when the answer needs to be refined with additional retrieved "
                "sources. Available variables: {existing_answer}, {context_msg}, "
                "{query_str}."
            ),
            variables=["existing_answer", "context_msg", "query_str"],
            default_text=(
                "Berikut adalah percakapan antara pengguna dan asisten AI. "
                "Asisten memberikan jawaban secara detail dan spesifik berdasarkan konteks yang diberikan. "
                "Jika asisten tidak mengetahui jawaban dari suatu pertanyaan, asisten akan menyatakan bahwa ia tidak mengetahuinya.\n\n"
                "Sebagai asisten, berikan jawaban hanya berdasarkan sumber-sumber yang diberikan dengan "
                "menyertakan sitasi pada paragraf. Saat mereferensikan informasi dari sebuah sumber, "
                "sebutkan sumber yang sesuai menggunakan ID masing-masing. "
                "Setiap jawaban atau paragraf harus menyertakan setidaknya satu sitasi sumber. "
                "Hanya buat sitasi jika Anda secara eksplisit mereferensikannya. "
                "Sitasi harus menggunakan tag anchor (<a class=\"rag_citation\" href=\"CITATION_HERE\"></a>) "
                "dan (SANGAT PENTING) diletakkan langsung di dalam teks (in-line). Jangan gunakan catatan kaki atau catatan akhir. "
                "Jika tidak ada sumber yang membantu, nyatakan hal tersebut. "
                "Jangan membuat ID sumber buatan. Hanya gunakan ID sumber yang tersedia pada konteks.\n\n"
                "Aturan Tambahan Jawaban:\n"
                "Di bagian paling akhir setiap jawaban, WAJIB buat tag XML <followups> yang berisi 4–5 opsi pertanyaan lanjutan yang interaktif, kontekstual, dan spesifik terkait data atau topik yang baru saja dijelaskan. Pisahkan setiap pertanyaan dengan karakter pipe (|). JANGAN gunakan list markdown di dalam tag ini.\n\n"
                "Format keluaran akhir:\n"
                "[Jawaban Anda di sini...]\n"
                "<followups>Pertanyaan drill-down spesifik?|Pertanyaan perbandingan produk/kategori?|Pertanyaan tren waktu?|Pertanyaan dampak/penyebab spesifik?</followups>\n\n"
                "Sekarang giliran Anda. Kami telah menyediakan jawaban yang sudah ada sebelumnya:\n\n"
                "<Existing Answer>\n"
                "{existing_answer}\n\n"
                "Di bawah ini adalah beberapa sumber informasi terhitung.\n"
                "Gunakan sumber tersebut untuk memperjelas jawaban yang ada.\n"
                "Jika sumber yang diberikan tidak membantu, ulangi kembali jawaban yang sudah ada.\n"
                "Mulai perjelas!\n\n"
                "<Contexts>\n"
                "{context_msg}\n\n"
                "<Query>\n"
                "{query_str}\n\n"
                "<Answer>\n"
            ),
        ),
        PromptDefinition(
            key=AGENT_SYSTEM_PROMPT,
            title="Agent System Prompt (Tool Calling)",
            description=(
                "System prompt for the tool-calling agent used in chat responses. "
                "Available variables: {date} (today's date), {time} (current time)."
            ),
            variables=["date", "time"],
            default_text='### DATE AND TIME\nToday\'s date is {date} and the current time is {time}. This date and time is considered the current date and time for all responses. \n### ROLE DESCRIPTION\nYou are an expert agent that can answer questions with the help of tools. You will use the date and time provided above to answer questions to refine the user\'s query and provide the best possible answer. \n### BEST PRACTICES\nYou will follow these best practices when answering questions:\n1. Refining the user\'s query according to the date and time provided above if necessary, and if the user \nhas not provided enough information to answer the question. 2. Going through the tools available.\n3. Approaching the question step by step, using the tools \navailable to you to gather information when necessary. \n4. Once you have the information you need, you will provide a final answer to the user with citations if available you used to answer the question.\n5. If you do not know the answer to a question or cannot find the information you need to answer the question with the provided sources or tools, you truthfully say you do not know and let the user know how you arrived at the response and what information you used (links if any) to arrive at it and ask for clarification or more information. \n\n### OUTPUT FORMAT\nAs the agent, you will provide an answer based solely on the provided sources with citations (if available). Only return the answer with citations (if used) to the user. If you cannot answer the question with the provided sources or tools, you will return a message saying you cannot answer the question and ask the user to provide more information or clarify the question. \n### CITATION FORMAT\nYou will use the following format to cite sources in your response:\n* Use the citations from the chat history as is.\n* Use links provided by tool results if needed to answer the question and cite them in-line in the given format: the link should be in markdown format. For example: Refer to the example in [example.com](https://example.com). Do not make up links that are not present.\n* Cite from tool results with node_ids in the given format: the node_id should be in an html anchor tag (<a href>) with an html \'class\' of \'rag_citation\'. Do not use filenames as citations. Only node ids should be used. For example: <a class="rag_citation" href="2" ></a>. Do not make up node ids that are not present \nin the context.\n* All citations should be either in-line citations or markdown links.\n\nFor example:\n\n<Contexts>\nSource: 1\nThe sky is red in the evening and blue in the morning.\n\nSource: 2\nWater is wet when the sky is red.\n\nSource: 3 \nwww.example1.com\nThe sky is red in the evening and blue in the morning.\n\nSource: 4 \nwww.example2.com\nOnly in the evenings, is the water wet.\n\n<Query>\nWhen is water wet?\n\n<Answer> \nWater will be wet when the sky is red<a class="rag_citation" href="1"></a> [example.com](www.example.com), which occurs in the evening. [example2](www.example2.com) <a class="rag_citation" href="2"></a>.\n',
        ),
        PromptDefinition(
            key=DOCUMENT_SUMMARY_PROMPT,
            title="Document Summary",
            description=(
                "Prompt used to generate summaries of documents when indexing a "
                "knowledge base. No variables available."
            ),
            variables=[],
            default_text=(
                "Buatlah ringkasan dari konten tersebut dalam kurang dari 100 kata dalam Bahasa Indonesia."
            ),
        ),
        PromptDefinition(
            key=OCR_PAGE_PROMPT,
            title="OCR Page",
            description=(
                "Prompt used when performing OCR of scanned document pages with the "
                "Qwen OCR model. No variables available."
            ),
            variables=[],
            default_text=(
                "Lakukan OCR pada satu halaman dokumen yang ditampilkan pada gambar. "
                "Kembalikan seluruh teks persis seperti yang terlihat, dengan mempertahankan tata letak sedapat mungkin. "
                "PENTING: Kembalikan setiap baris tabel dan setiap item daftar secara lengkap dan persis seperti aslinya; jangan "
                "merangkum, memotong, atau menghilangkan baris tabel apa pun. "
                "Jika halaman berisi grafik atau diagram, berikan ringkasan singkat secara terpisah di bagian akhir, "
                "dengan label 'RINGKASAN GRAFIK:'. "
                "Jangan merangkum atau menghilangkan data tabel apa pun. "
                "Jangan memformat output sebagai HTML atau XML; WAJIB mengembalikan plain text."
            ),
        ),
        PromptDefinition(
            key=HYDE_PROMPT,
            title="Hypothetical Document (HyDE)",
            description=(
                "Used when the HyDE query transformation is enabled in chat settings. "
                "Available variables: {question}."
            ),
            variables=["question"],
            default_text=(
                "Anda adalah seorang ahli di bidang terkait. Anda diminta untuk menjawab pertanyaan berikut: {question}. "
                "Buatlah dokumen singkat yang secara hipotetis dapat memberikan jawaban atas pertanyaan tersebut."
            ),
        ),
        PromptDefinition(
            key=SESSION_RENAME_PROMPT,
            title="Session Renaming",
            description=(
                "Used to suggest a name for a new chat session based on the first "
                "interaction. Available variables: {user_message}, {assistant_message}."
            ),
            variables=["user_message", "assistant_message"],
            default_text=(
                "You are tasked with suggesting an apt name for a chat session based on its first interaction between a User and an Assistant. \n\n"
                "# Instructions\n"
                "IMPORTANTLY, ONLY RETURN THE NAME OF THE SESSION.  Only return a single line and sessions name, without any additional text or formatting.\n"
                "Use the below interactions as a guide but do not include them in your response.\n\n"
                "### Example 1:\n"
                "First Interaction:\n"
                "```\n"
                "User: What is your name?\n"
                "Assistant: My name is Assistant.\n"
                "```\n\n"
                "Session Name:\n"
                "Introduction\n\n"
                "### Example 2:\n"
                "First Interaction:\n"
                "```\n"
                "User: What do you know about the Moon?\n"
                "Assistant: The Moon is Earth's only natural satellite. It is the fifth-largest satellite in the Solar System, and by far the largest among planetary satellites relative to the size of the planet that it orbits.\n"
                "```\n\n"
                "Session Name:\n"
                "Facts about the Moon\n\n"
                "# Your turn:\n"
                "First Interaction:\n"
                "```\n"
                "User: {user_message}\n"
                "Assistant: {assistant_message}\n"
                "```\n\n"
                "Session Name: \n"
            ),
        ),
        PromptDefinition(
            key=SUGGESTED_QUESTIONS_DIRECT_PROMPT,
            title="Suggested Questions",
            description=(
                "Used to generate suggested follow-up questions for sessions that "
                "have no indexed knowledge-base content (direct LLM mode). No "
                "variables available."
            ),
            variables=[],
            default_text=(
                " Berikan daftar pertanyaan lanjutan yang mungkin relevan."
                " Setiap pertanyaan harus ditulis pada baris baru."
                " Tidak boleh ada lebih dari empat (4) pertanyaan."
                " Setiap pertanyaan tidak boleh lebih dari lima belas (15) kata."
                " Respons harus berupa daftar bullet, dengan tanda bintang (*) sebagai bullet."
                " Jangan memulai respons seperti ini - Berikut adalah empat pertanyaan yang dapat saya jawab berdasarkan informasi yang tersedia"
                " Hanya kembalikan daftar pertanyaan."
                " Hanya gunakan plain text."
                " Jangan gunakan tag HTML atau format Markdown."
            ),
        ),
    ]
}


def prompts_dir() -> str:
    return settings.prompts_dir


def _overrides_file_path() -> str:
    return os.path.join(prompts_dir(), PROMPT_OVERRIDES_FILE)


def load_prompt_overrides() -> Dict[str, str]:
    if not os.path.exists(_overrides_file_path()):
        return {}
    try:
        with open(_overrides_file_path(), "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {k: str(v) for k, v in data.items() if k in PROMPT_DEFINITIONS}
        return {}
    except Exception:
        return {}


def save_prompt_overrides(overrides: Dict[str, str]) -> None:
    os.makedirs(prompts_dir(), exist_ok=True)
    with open(_overrides_file_path(), "w") as f:
        json.dump(overrides, f, indent=2)


def get_prompt(prompt_key: str) -> str:
    """Resolve a prompt: override if set, otherwise the default."""
    overrides = load_prompt_overrides()
    if prompt_key in overrides and overrides[prompt_key].strip():
        return overrides[prompt_key]
    return PROMPT_DEFINITIONS[prompt_key].default_text


def get_all_prompts() -> List[Dict[str, Any]]:
    overrides = load_prompt_overrides()
    result: List[Dict[str, Any]] = []
    for definition in PROMPT_DEFINITIONS.values():
        override = overrides.get(definition.key)
        result.append(
            {
                "key": definition.key,
                "title": definition.title,
                "description": definition.description,
                "variables": definition.variables,
                "default_text": definition.default_text,
                "text": override if override is not None else definition.default_text,
                "is_modified": bool(override is not None and override.strip()),
            }
        )
    return result


def save_prompts(new_values: Dict[str, str]) -> None:
    unknown = [key for key in new_values if key not in PROMPT_DEFINITIONS]
    if unknown:
        raise ValueError(f"Unknown prompt keys: {unknown}")
    overrides = load_prompt_overrides()
    for key, value in new_values.items():
        if value.strip():
            overrides[key] = value
        else:
            overrides.pop(key, None)
    save_prompt_overrides(overrides)


def reset_prompts(prompt_keys: Optional[List[str]] = None) -> None:
    overrides = load_prompt_overrides()
    if prompt_keys is None:
        overrides = {}
    else:
        for key in prompt_keys:
            overrides.pop(key, None)
    save_prompt_overrides(overrides)
