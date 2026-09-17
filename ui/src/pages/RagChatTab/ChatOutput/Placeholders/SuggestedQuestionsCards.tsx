/*******************************************************************************
 * CLOUDERA APPLIED MACHINE LEARNING PROTOTYPE (AMP)
 * (C) Cloudera, Inc. 2024
 * All rights reserved.
 *
 * Applicable Open Source License: Apache 2.0
 *
 * NOTE: Cloudera open source products are modular software products
 * made up of hundreds of individual components, each of which was
 * individually copyrighted.  Each Cloudera open source product is a
 * collective work under U.S. Copyright Law. Your license to use the
 * collective work is as provided in your written agreement with
 * Cloudera.  Used apart from the collective work, this file is
 * licensed for your use pursuant to the open source license
 * identified above.
 *
 * This code is provided to you pursuant a written agreement with
 * (i) Cloudera, Inc. or (ii) a third-party authorized to distribute
 * this code. If you do not have a written agreement with Cloudera nor
 * with an authorized and properly licensed third party, you do not
 * have any rights to access nor to use this code.
 *
 * Absent a written agreement with Cloudera, Inc. (“Cloudera”) to the
 * contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
 * KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED
 * WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO
 * IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU,
 * AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS
 * ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE
 * OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
 * CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES
 * RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF
 * BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
 * DATA.
 ******************************************************************************/

import { Card, Flex, Skeleton, Typography } from "antd";
import { RagChatContext } from "pages/RagChatTab/State/RagChatContext.tsx";
import { useContext } from "react";
// import { useSuggestQuestions } from "src/api/ragQueryApi.ts";
import {
  createQueryConfiguration,
  getOnEvent,
  useStreamingChatMutation,
} from "src/api/chatApi.ts";
import useCreateSessionAndRedirect from "pages/RagChatTab/ChatOutput/hooks/useCreateSessionAndRedirect";
import { useStreamingChunkBuffer } from "src/hooks/useStreamingChunkBuffer.ts";

const QuestionCard = ({
  question,
  index,
  onClick,
}: {
  question: string;
  index: number;
  onClick: (suggestedQuestion: string) => void;
}) => {
  return (
    <Card
      size={"small"}
      hoverable
      style={{
        width: 178,
        margin: 0,
        padding: 0,
      }}
      extra={
        <Typography.Text type="secondary">{`#${(index + 1).toString()}`}</Typography.Text>
      }
      onClick={() => {
        onClick(question);
      }}
    >
      <Card.Meta description={<Typography.Text>{question}</Typography.Text>} />
    </Card>
  );
};

const SuggestedQuestionsCards = () => {
  const {
    activeSession,
    excludeKnowledgeBaseState: [excludeKnowledgeBase],
    streamedChatState: [, setStreamedChat],
    streamedEventState: [, setStreamedEvent],
    streamedAbortControllerState: [, setStreamedAbortController],
    // 1. Pull our new fast follow-ups from the context
    streamedFollowupsState: [streamedFollowups],
  } = useContext(RagChatContext);
  
  const sessionId = activeSession?.id;

  const createSessionAndRedirect = useCreateSessionAndRedirect();

  // Track whether we are currently inside a <followups>...</followups> block
  const isBufferingFollowups = useRef(false);
  const followupsBuffer = useRef("");

  // Use custom hook to handle batched streaming updates
  const { onChunk, flush } = useStreamingChunkBuffer((chunks) => {
    // Check if the tag is starting
    if (chunks.includes("<followups>")) {
      isBufferingFollowups.current = true;
      const parts = chunks.split("<followups>");
      if (parts[0]) setStreamedChat((prev) => prev + parts[0]);
      followupsBuffer.current += parts[1] || "";
      return;
    }

    // Check if the tag is ending
    if (chunks.includes("</followups>")) {
      isBufferingFollowups.current = false;
      const parts = chunks.split("</followups>");
      followupsBuffer.current += parts[0] || "";

      // Parse the pipe-separated string into an array and update state
      const questions = followupsBuffer.current
        .split("|")
        .map((q) => q.trim())
        .filter(Boolean);
      setStreamedFollowups(questions);

      // Reset the buffer for the next chat
      followupsBuffer.current = "";
      return;
    }

    // Route the chunk to the correct destination
    if (isBufferingFollowups.current) {
      followupsBuffer.current += chunks;
    } else {
      setStreamedChat((prev) => prev + chunks);
    }
  });

  const { mutate: chatMutation, isPending: askRagIsPending } =
    useStreamingChatMutation({
      onChunk,
      onEvent: getOnEvent(setStreamedEvent),
      onSuccess: () => {
        // Flush any remaining chunks before cleanup
        flush();
        setStreamedChat("");
      },
      getController: (ctrl: AbortController) => {
        setStreamedAbortController(ctrl);
      },
    });

  const handleAskSample = (suggestedQuestion: string) => {
    if (suggestedQuestion.length > 0) {
      if (sessionId) {
        chatMutation({
          query: suggestedQuestion,
          session_id: sessionId,
          configuration: createQueryConfiguration(excludeKnowledgeBase),
        });
      } else {
        createSessionAndRedirect([], suggestedQuestion);
      }
    }
  };

  // 2. We only show the loading skeletons while the AI is actively generating the response
  if (askRagIsPending) {
    return (
      <Flex gap={10} wrap="wrap" justify="space-between">
        {Array.from({ length: 4 }).map((_, index) => (
          <Skeleton
            key={index}
            active
            style={{
              width: 178,
              padding: 0,
            }}
          />
        ))}
      </Flex>
    );
  }

  // 3. Map over the fast streamed follow-ups instead of the old data.suggested_questions
  return (
    <Flex gap={10} wrap="wrap" justify="space-between">
      {streamedFollowups.map((question, index) => {
        return (
          <QuestionCard
            question={question}
            index={index}
            key={index}
            onClick={handleAskSample}
          />
        );
      })}
    </Flex>
  );
};

export default SuggestedQuestionsCards;
