/*******************************************************************************
 * CLOUDERA APPLIED MACHINE LEARNING PROTOTYPE (AMP)
 * (C) Cloudera, Inc. 2025
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

import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  RagChatContext,
  RagChatContextType,
} from "pages/RagChatTab/State/RagChatContext.tsx";
import SuggestedQuestionsCards from "./SuggestedQuestionsCards";

const STORAGE_KEY = "suggestedQuestionsCollapsed";

const createMockContext = (
  overrides: Partial<RagChatContextType> = {},
): RagChatContextType => ({
  activeSession: undefined,
  chatHistoryQuery: {
    flatChatHistory: [],
    isFetching: false,
    isFetchingPreviousPage: false,
    chatHistoryStatus: "success",
    fetchPreviousPage: vi.fn(),
  },
  streamedChatState: ["", vi.fn()],
  streamedEventState: [[], vi.fn()],
  streamedAbortControllerState: [undefined, vi.fn()],
  streamedFollowupsState: [[], vi.fn()],
  draftQuestionState: ["", vi.fn()],
  dataSourcesQuery: { dataSources: [], dataSourcesStatus: "success" },
  dataSourceSize: null,
  excludeKnowledgeBaseState: [false, vi.fn()],
  ...overrides,
});

const renderCards = (contextValue: RagChatContextType) =>
  render(
    <RagChatContext.Provider value={contextValue}>
      <SuggestedQuestionsCards />
    </RagChatContext.Provider>,
  );

describe("SuggestedQuestionsCards", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  afterEach(() => {
    cleanup();
    localStorage.clear();
  });

  it("renders at most five suggested questions when expanded", () => {
    renderCards(
      createMockContext({
        streamedFollowupsState: [
          [
            "Question 1",
            "Question 2",
            "Question 3",
            "Question 4",
            "Question 5",
            "Question 6",
          ],
          vi.fn(),
        ],
      }),
    );

    expect(screen.getByText("Question 1")).toBeInTheDocument();
    expect(screen.getByText("Question 5")).toBeInTheDocument();
    expect(screen.queryByText("Question 6")).not.toBeInTheDocument();
  });

  it("places the clicked suggestion in the chat input instead of asking it", async () => {
    const user = userEvent.setup();
    const setDraftQuestion = vi.fn();
    renderCards(
      createMockContext({
        streamedFollowupsState: [["A question to edit"], vi.fn()],
        draftQuestionState: ["", setDraftQuestion],
      }),
    );

    await user.click(screen.getByText("A question to edit"));

    expect(setDraftQuestion).toHaveBeenCalledWith("A question to edit");
  });

  it("hides the cards when the suggestions are minimized", async () => {
    const user = userEvent.setup();
    renderCards(
      createMockContext({
        streamedFollowupsState: [["A question to edit"], vi.fn()],
      }),
    );

    await user.click(screen.getByTestId("suggested-questions-toggle"));

    expect(screen.queryByText("A question to edit")).not.toBeInTheDocument();
    expect(
      screen.getByText("Suggested Follow-up Questions"),
    ).toBeInTheDocument();
    expect(localStorage.getItem(STORAGE_KEY)).toBe("true");
  });
});
