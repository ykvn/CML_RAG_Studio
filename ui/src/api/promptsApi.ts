/*
 * CLOUDERA APPLIED MACHINE LEARNING PROTOTYPE (AMP)
 * (C) Cloudera, Inc. 2026
 * All rights reserved.
 *
 * Applicable Open Source License: Apache 2.0
 */

import { queryOptions, useMutation, useQuery } from "@tanstack/react-query";
import {
  commonHeaders,
  CustomError,
  llmServicePath,
  MutationKeys,
  QueryKeys,
  UseMutationType,
} from "src/api/utils.ts";

export interface Prompt {
  key: string;
  title: string;
  description: string;
  variables: string[];
  default_text: string;
  text: string;
  is_modified: boolean;
}

export interface UpdatePromptsRequest {
  prompts: Record<string, string>;
}

const putRequest = async <T>(
  url: string,
  body: UpdatePromptsRequest | { keys?: string[] | null },
): Promise<T> => {
  const res = await fetch(url, {
    method: "PUT",
    body: JSON.stringify(body),
    headers: {
      ...commonHeaders,
      "Content-Type": "application/json",
    },
  });
  if (!res.ok) {
    const detail = (await res.json()) as CustomError;
    throw new Error(detail.message ?? detail.detail ?? res.statusText);
  }
  return await ((await res.json()) as Promise<T>);
};

const postRequest = async <T>(
  url: string,
  body: UpdatePromptsRequest | { keys?: string[] | null },
): Promise<T> => {
  const res = await fetch(url, {
    method: "POST",
    body: JSON.stringify(body),
    headers: {
      ...commonHeaders,
      "Content-Type": "application/json",
    },
  });
  if (!res.ok) {
    const detail = (await res.json()) as CustomError;
    throw new Error(detail.message ?? detail.detail ?? res.statusText);
  }
  return await ((await res.json()) as Promise<T>);
};

const getPrompts = async (): Promise<Prompt[]> => {
  const res = await fetch(`${llmServicePath}/prompts`, {
    method: "GET",
    headers: { ...commonHeaders },
  });
  if (!res.ok) {
    const detail = (await res.json()) as CustomError;
    throw new Error(detail.message ?? detail.detail ?? res.statusText);
  }
  return await ((await res.json()) as Promise<Prompt[]>);
};

export const getPromptsQueryOptions = queryOptions({
  queryKey: [QueryKeys.getPrompts],
  queryFn: getPrompts,
});

export const useGetPrompts = () => {
  return useQuery(getPromptsQueryOptions);
};

export const useUpdatePrompts = ({
  onSuccess,
  onError,
}: UseMutationType<null>) => {
  return useMutation({
    mutationKey: [MutationKeys.updatePrompts],
    mutationFn: async (request: UpdatePromptsRequest) =>
      putRequest<null>(`${llmServicePath}/prompts`, request),
    onSuccess: onSuccess,
    onError: onError,
  });
};

export const useResetPrompts = ({
  onSuccess,
  onError,
}: UseMutationType<null>) => {
  return useMutation({
    mutationKey: [MutationKeys.resetPrompts],
    mutationFn: async (keys: string[] | null) =>
      postRequest<null>(`${llmServicePath}/prompts/reset`, { keys: keys }),
    onSuccess: onSuccess,
    onError: onError,
  });
};
