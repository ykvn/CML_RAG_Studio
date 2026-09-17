import { renderHook } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { useFollowupsStreamParser } from "src/hooks/useFollowupsStreamParser.ts";

const renderParser = () => {
  const setStreamedChat = vi.fn();
  const setStreamedFollowups = vi.fn();
  const hook = renderHook(() =>
    useFollowupsStreamParser(setStreamedChat, setStreamedFollowups),
  );
  return {
    setStreamedChat,
    setStreamedFollowups,
    push: (chunk: string) => {
      hook.result.current.onChunk(chunk);
      hook.result.current.flush();
    },
    reset: () => {
      hook.result.current.reset();
    },
  };
};

describe("useFollowupsStreamParser", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("parses a complete followups block in a single chunk", () => {
    const parser = renderParser();
    parser.push("Answer text<followups>Q1?|Q2?</followups>");

    expect(parser.setStreamedChat).toHaveBeenCalledWith("Answer text");
    expect(parser.setStreamedFollowups).toHaveBeenCalledWith(["Q1?", "Q2?"]);
  });

  it("handles the open tag split across chunk boundaries", () => {
    const parser = renderParser();
    parser.push("Answer<fol");
    parser.push("lowups>Q1?|Q2?|Q3?</fol");
    parser.push("lowups>");

    const chatCalls = parser.setStreamedChat.mock.calls.map(
      (call) => call[0] as string,
    );
    // The tag must never leak into the chat text
    chatCalls.forEach((text) => {
      expect(text).not.toContain("<");
    });
    expect(chatCalls[chatCalls.length - 1]).toBe("Answer");
    expect(parser.setStreamedFollowups).toHaveBeenCalledWith([
      "Q1?",
      "Q2?",
      "Q3?",
    ]);
  });

  it("handles the close tag split across chunk boundaries", () => {
    const parser = renderParser();
    parser.push("Answer<followups>Q1?|Q2?</follow");
    parser.push("ups>");

    expect(parser.setStreamedChat).toHaveBeenCalledWith("Answer");
    expect(parser.setStreamedFollowups).toHaveBeenCalledWith(["Q1?", "Q2?"]);
  });

  it("holds back a trailing partial open tag until resolved", () => {
    const parser = renderParser();
    parser.push("Hello <f");

    expect(parser.setStreamedChat).toHaveBeenLastCalledWith("Hello ");
    expect(parser.setStreamedFollowups).not.toHaveBeenCalled();

    parser.push("ollowups>Q1?</followups>");
    expect(parser.setStreamedChat).toHaveBeenLastCalledWith("Hello ");
    expect(parser.setStreamedFollowups).toHaveBeenCalledWith(["Q1?"]);
  });

  it("passes plain text through without tags", () => {
    const parser = renderParser();
    parser.push("just a normal answer");

    expect(parser.setStreamedChat).toHaveBeenCalledWith("just a normal answer");
    expect(parser.setStreamedFollowups).not.toHaveBeenCalled();
  });

  it("strips followups from the middle of the text", () => {
    const parser = renderParser();
    parser.push("Before<followups>Q1?</followups>After");

    expect(parser.setStreamedChat).toHaveBeenLastCalledWith("Before");
    expect(parser.setStreamedFollowups).toHaveBeenCalledWith(["Q1?"]);
  });

  it("reset clears state between chats", () => {
    const parser = renderParser();
    parser.push("Answer<followups>Q1?</followups>");
    parser.reset();

    expect(parser.setStreamedFollowups).toHaveBeenLastCalledWith([]);

    // A new plain-text stream is unaffected by the previous tag state
    parser.push("fresh answer");
    expect(parser.setStreamedChat).toHaveBeenLastCalledWith("fresh answer");
  });
});
