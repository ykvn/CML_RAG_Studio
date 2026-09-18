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
import SuggestedQuestionsFooter from "./SuggestedQuestionsFooter";

const STORAGE_KEY = "suggestedQuestionsCollapsed";

const questions = [
  "Question 1",
  "Question 2",
  "Question 3",
  "Question 4",
  "Question 5",
  "Question 6",
];

const renderFooter = (handleChat = vi.fn()) => {
  render(
    <SuggestedQuestionsFooter
      isLoading={false}
      handleChat={handleChat}
      questions={questions}
      error={null}
    />,
  );
  return { handleChat };
};

describe("SuggestedQuestionsFooter", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  afterEach(() => {
    cleanup();
    localStorage.clear();
  });

  it("renders at most five suggested questions when expanded", () => {
    renderFooter();

    expect(screen.getByText("Question 1")).toBeInTheDocument();
    expect(screen.getByText("Question 5")).toBeInTheDocument();
    expect(screen.queryByText("Question 6")).not.toBeInTheDocument();
    expect(screen.getByTestId("suggested-questions-toggle")).toHaveAttribute(
      "aria-expanded",
      "true",
    );
  });

  it("minimizes and restores the suggested questions", async () => {
    const user = userEvent.setup();
    renderFooter();

    const toggle = screen.getByTestId("suggested-questions-toggle");

    await user.click(toggle);

    expect(screen.queryByText("Question 1")).not.toBeInTheDocument();
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    expect(localStorage.getItem(STORAGE_KEY)).toBe("true");

    await user.click(toggle);

    expect(screen.getByText("Question 1")).toBeInTheDocument();
    expect(toggle).toHaveAttribute("aria-expanded", "true");
    expect(localStorage.getItem(STORAGE_KEY)).toBe("false");
  });

  it("starts minimized when the preference was persisted", () => {
    localStorage.setItem(STORAGE_KEY, "true");

    renderFooter();

    expect(
      screen.getByText("Suggested Follow-up Questions"),
    ).toBeInTheDocument();
    expect(screen.queryByText("Question 1")).not.toBeInTheDocument();
    expect(screen.getByTestId("suggested-questions-toggle")).toHaveAttribute(
      "aria-expanded",
      "false",
    );
  });

  it("hides the loading skeleton while minimized", () => {
    localStorage.setItem(STORAGE_KEY, "true");

    render(
      <SuggestedQuestionsFooter
        isLoading={true}
        handleChat={vi.fn()}
        questions={[]}
        error={null}
      />,
    );

    expect(document.querySelector(".ant-skeleton")).toBeNull();
  });

  it("submits the selected suggested question", async () => {
    const user = userEvent.setup();
    const handleChat = vi.fn();
    renderFooter(handleChat);

    await user.click(screen.getByText("Question 2"));

    expect(handleChat).toHaveBeenCalledWith("Question 2");
  });
});
