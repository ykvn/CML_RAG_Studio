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
import concurrent.futures
import logging
from typing import List

from llama_index.core import QueryBundle
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore

from app.services.query.flexible_retriever import FlexibleRetriever

logger = logging.getLogger(__name__)


class MultiSourceRetriever(BaseRetriever):
    def __init__(self, retrievers: list[FlexibleRetriever]):
        super().__init__()
        self.retrievers = retrievers

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        # Each source's retrieval is dominated by network round-trips (an LLM
        # query-embedding call plus one or more Qdrant searches), and the
        # per-source stores are independent (separate collections/clients).
        # Run them concurrently to cut total retrieval time roughly in half.
        if not self.retrievers:
            return []
        results: list[NodeWithScore] = []
        # Bounded, small pool to avoid saturating the gateway/vector store.
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(len(self.retrievers), 4)
        ) as executor:
            futures = [
                executor.submit(retriever.retrieve, query_bundle)
                for retriever in self.retrievers
            ]
            for future in futures:
                try:
                    results.extend(future.result())
                except Exception:
                    logger.exception("Failed to retrieve from one data source")
        return results


