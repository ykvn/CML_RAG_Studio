import { useCallback, useRef } from "react";
import type { Dispatch, SetStateAction } from "react";
import { useStreamingChunkBuffer } from "src/hooks/useStreamingChunkBuffer.ts";

const OPEN_TAG = "<followups>";
const CLOSE_TAG = "</followups>";

const parseQuestions = (raw: string): string[] =>
  raw
    .split("|")
    .map((q) => q.trim())
    .filter(Boolean);

/**
 * Returns the length of a trailing partial OPEN_TAG (e.g. "<follo") so it can
 * be held back from the rendered chat text until more chunks arrive.
 */
const partialOpenTagLength = (text: string): number => {
  const max = Math.min(text.length, OPEN_TAG.length - 1);
  for (let i = max; i > 0; i--) {
    if (text.endsWith(OPEN_TAG.slice(0, i))) {
      return i;
    }
  }
  return 0;
};

/**
 * Robust streaming parser for <followups>...</followups> blocks.
 *
 * Unlike per-chunk tag matching, this accumulates the FULL streamed text in a
 * ref and re-parses the whole text on every batched flush, so tags that are
 * split across chunk/network boundaries are handled correctly.
 *
 * - Complete blocks: extracts pipe-separated questions into
 *   `setStreamedFollowups` and strips the block from the chat text.
 * - Incomplete open tag: holds the text after it back from the chat display.
 * - Partial trailing open tag (e.g. "<foll"): held back until resolved.
 *
 * Call `reset()` when a new chat starts to clear accumulated state.
 */
export const useFollowupsStreamParser = (
  setStreamedChat: Dispatch<SetStateAction<string>>,
  setStreamedFollowups: Dispatch<SetStateAction<string[]>>,
) => {
  const fullTextRef = useRef("");

  const reset = useCallback(() => {
    fullTextRef.current = "";
    setStreamedFollowups([]);
  }, [setStreamedFollowups]);

  const processText = useCallback(() => {
    const text = fullTextRef.current;
    const openIdx = text.lastIndexOf(OPEN_TAG);

    if (openIdx !== -1) {
      const closeIdx = text.indexOf(CLOSE_TAG, openIdx);
      // Display only the text before the open tag
      setStreamedChat(text.slice(0, openIdx));
      if (closeIdx !== -1) {
        // Complete block: extract the questions
        const content = text.slice(openIdx + OPEN_TAG.length, closeIdx);
        setStreamedFollowups(parseQuestions(content));
      }
      return;
    }

    // No open tag — hold back any trailing partial open tag from the display
    const holdBack = partialOpenTagLength(text);
    setStreamedChat(text.slice(0, text.length - holdBack));
  }, [setStreamedChat, setStreamedFollowups]);

  const { onChunk, flush } = useStreamingChunkBuffer((chunks) => {
    fullTextRef.current += chunks;
    processText();
  });

  return { onChunk, flush, reset };
};
