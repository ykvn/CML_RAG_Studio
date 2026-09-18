import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { useSuggestedQuestionsCollapsed } from "src/hooks/useSuggestedQuestionsCollapsed.ts";

const STORAGE_KEY = "suggestedQuestionsCollapsed";

describe("useSuggestedQuestionsCollapsed", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  afterEach(() => {
    localStorage.clear();
  });

  it("defaults to expanded when no preference is stored", () => {
    const { result } = renderHook(() => useSuggestedQuestionsCollapsed());

    expect(result.current.collapsed).toBe(false);
  });

  it("toggles the collapsed state and persists the preference", () => {
    const { result } = renderHook(() => useSuggestedQuestionsCollapsed());

    act(() => {
      result.current.toggleCollapsed();
    });

    expect(result.current.collapsed).toBe(true);
    expect(localStorage.getItem(STORAGE_KEY)).toBe("true");

    act(() => {
      result.current.toggleCollapsed();
    });

    expect(result.current.collapsed).toBe(false);
    expect(localStorage.getItem(STORAGE_KEY)).toBe("false");
  });

  it("restores the persisted preference on mount", () => {
    localStorage.setItem(STORAGE_KEY, "true");

    const { result } = renderHook(() => useSuggestedQuestionsCollapsed());

    expect(result.current.collapsed).toBe(true);
  });
});
