import { useCallback, useState } from "react";

const SUGGESTED_QUESTIONS_COLLAPSED_STORAGE_KEY = "suggestedQuestionsCollapsed";

const getStoredCollapsedState = (): boolean =>
  localStorage.getItem(SUGGESTED_QUESTIONS_COLLAPSED_STORAGE_KEY) === "true";

/**
 * Shared, persisted collapse state for the suggested follow-up questions.
 *
 * Used by both the chat footer (shown after messages) and the empty-chat
 * suggestion cards so a single user preference minimizes both.
 */
export const useSuggestedQuestionsCollapsed = () => {
  const [collapsed, setCollapsed] = useState(getStoredCollapsedState);

  const toggleCollapsed = useCallback(() => {
    setCollapsed((previous) => {
      const nextCollapsed = !previous;
      localStorage.setItem(
        SUGGESTED_QUESTIONS_COLLAPSED_STORAGE_KEY,
        String(nextCollapsed),
      );
      return nextCollapsed;
    });
  }, []);

  return { collapsed, toggleCollapsed };
};
