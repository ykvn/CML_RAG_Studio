import { describe, expect, it } from "vitest";
import {
  filterChatSelectableModels,
  transformChatModelOptions,
  transformModelOptions,
} from "src/utils/modelUtils.ts";
import { Model } from "src/api/modelsApi.ts";

const makeModel = (
  model_id: string,
  overrides: Partial<Model> = {},
): Model => ({
  name: model_id,
  model_id,
  available: true,
  tool_calling_supported: true,
  ...overrides,
});

const llmModels: Model[] = [
  makeModel("Qwen3.8-27B-no-thinking"),
  makeModel("Qwen3.8-27B-low"),
  makeModel("Qwen3.8-27B-medium"),
  makeModel("Qwen3.8-27B-ocr"),
  makeModel("bge-m3", { tool_calling_supported: false }),
  makeModel("bge-reranker-v2-m3", { tool_calling_supported: false }),
];

describe("filterChatSelectableModels", () => {
  it("removes non-chat (OCR) models from the list", () => {
    const result = filterChatSelectableModels(llmModels);
    expect(result.map((model) => model.model_id)).toEqual([
      "Qwen3.8-27B-no-thinking",
      "Qwen3.8-27B-low",
      "Qwen3.8-27B-medium",
      "bge-m3",
      "bge-reranker-v2-m3",
    ]);
  });

  it("keeps every model when no non-chat model is present", () => {
    const models = [makeModel("Qwen3.8-27B-low"), makeModel("bge-m3")];
    expect(filterChatSelectableModels(models)).toEqual(models);
  });

  it("returns an empty array for undefined input", () => {
    expect(filterChatSelectableModels(undefined)).toEqual([]);
  });

  it("returns an empty array when the list only contains non-chat models", () => {
    expect(filterChatSelectableModels([makeModel("Qwen3.8-27B-ocr")])).toEqual(
      [],
    );
  });
});

describe("transformChatModelOptions", () => {
  it("maps models to Select options without the non-chat models", () => {
    expect(transformChatModelOptions(llmModels)).toEqual([
      { value: "Qwen3.8-27B-no-thinking", label: "Qwen3.8-27B-no-thinking" },
      { value: "Qwen3.8-27B-low", label: "Qwen3.8-27B-low" },
      { value: "Qwen3.8-27B-medium", label: "Qwen3.8-27B-medium" },
      { value: "bge-m3", label: "bge-m3" },
      { value: "bge-reranker-v2-m3", label: "bge-reranker-v2-m3" },
    ]);
  });

  it("returns an empty array for undefined input", () => {
    expect(transformChatModelOptions(undefined)).toEqual([]);
  });
});

describe("transformModelOptions", () => {
  it("still lists every model, including non-chat (OCR) models", () => {
    expect(transformModelOptions(llmModels).map((option) => option.value)).toContain(
      "Qwen3.8-27B-ocr",
    );
  });
});
