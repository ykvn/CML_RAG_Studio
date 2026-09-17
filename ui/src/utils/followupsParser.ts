/**
 * Parses <followups> tags from a streamed response chunk.
 * The LLM emits follow-up questions in the format:
 * <followups>Question 1?|Question 2?|Question 3?</followups>
 *
 * Returns an array of extracted follow-up question strings, or null if
 * no complete <followups> block is found yet.
 */
export function parseFollowupsFromChunk(chunk: string): string[] | null {
  const followupsRegex = /<followups>([^<]*)<\/followups>/;
  const match = chunk.match(followupsRegex);
  if (!match) {
    return null;
  }
  const raw = match[1].trim();
  if (!raw) {
    return [];
  }
  return raw.split("|").map((q) => q.trim()).filter(Boolean);
}

/**
 * Extracts the "clean" text by stripping <followups>...</followups> blocks.
 * This is used so the follow-up tag content doesn't appear in the chat output.
 */
export function stripFollowupsFromText(text: string): string {
  return text.replace(/<followups>[^<]*<\/followups>/g, "").trim();
}
