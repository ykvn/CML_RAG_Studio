/*
 * CLOUDERA APPLIED MACHINE LEARNING PROTOTYPE (AMP)
 * (C) Cloudera, Inc. 2026
 * All rights reserved.
 *
 * Applicable Open Source License: Apache 2.0
 */

import { useMemo, useState } from "react";
import {
  Alert,
  Button,
  Collapse,
  Flex,
  Input,
  Popconfirm,
  Space,
  Tag,
  Typography,
} from "antd";
import { useQueryClient } from "@tanstack/react-query";
import messageQueue from "src/utils/messageQueue.ts";
import {
  Prompt,
  useGetPrompts,
  useResetPrompts,
  useUpdatePrompts,
} from "src/api/promptsApi.ts";
import { QueryKeys } from "src/api/utils.ts";

const ModelPromptPage = () => {
  const queryClient = useQueryClient();
  const { data: prompts, isLoading, error } = useGetPrompts();
  const [edited, setEdited] = useState<Record<string, string>>({});

  const onSuccess = async () => {
    messageQueue.success("Prompts saved successfully.");
    await queryClient.invalidateQueries({ queryKey: [QueryKeys.getPrompts] });
    setEdited({});
  };

  const onResetSuccess = async () => {
    messageQueue.success("Prompts reset to defaults.");
    await queryClient.invalidateQueries({ queryKey: [QueryKeys.getPrompts] });
    setEdited({});
  };

  const onError = (e: Error) => {
    messageQueue.error(
      e.message || "An error occurred while saving the prompts.",
    );
  };

  const updatePromptsMutation = useUpdatePrompts({
    onSuccess: onSuccess,
    onError: onError,
  });
  const resetPromptsMutation = useResetPrompts({
    onSuccess: onResetSuccess,
    onError: onError,
  });

  const dirtyCount = useMemo(
    () =>
      prompts?.filter(
        (p) => p.key in edited && edited[p.key] !== p.text,
      ).length ?? 0,
    [edited, prompts],
  );

  if (error) {
    return (
      <Alert
        type="error"
        message="Failed to load prompts"
        description={error.message}
      />
    );
  }

  const handleChange = (key: string, value: string) => {
    setEdited((prev) => ({ ...prev, [key]: value }));
  };

  const handleSave = () => {
    const changed: Record<string, string> = {};
    prompts?.forEach((p: Prompt) => {
      if (p.key in edited && edited[p.key] !== p.text) {
        changed[p.key] = edited[p.key];
      }
    });
    if (Object.keys(changed).length === 0) {
      messageQueue.warning("No prompt changes to save.");
      return;
    }
    updatePromptsMutation.mutate({ prompts: changed });
  };

  const handleReset = (key?: string) => {
    resetPromptsMutation.mutate(key ? [key] : null);
  };

  const items = (prompts ?? []).map((p: Prompt, index: number) => {
    const text = edited[p.key] ?? p.text;
    const modified = p.key in edited ? text !== p.text : p.is_modified;
    return {
      key: p.key,
      label: (
        <Space>
          <Typography.Text strong>
            {index + 1}. {p.title}
          </Typography.Text>
          {modified ? <Tag color="orange">Modified</Tag> : null}
        </Space>
      ),
      extra: (
        <Popconfirm
          title="Reset this prompt to its default?"
          onConfirm={(e) => {
            e?.stopPropagation();
            handleReset(p.key);
          }}
          onCancel={(e) => {
            e?.stopPropagation();
          }}
        >
          <Button
            size="small"
            type="link"
            onClick={(e) => {
              e.stopPropagation();
            }}
            disabled={resetPromptsMutation.isPending}
          >
            Reset
          </Button>
        </Popconfirm>
      ),
      children: (
        <Flex vertical gap={12}>
          <Typography.Paragraph type="secondary" style={{ marginBottom: 0 }}>
            {p.description}
          </Typography.Paragraph>
          {p.variables.length > 0 ? (
            <Space size={4} wrap>
              <Typography.Text type="secondary">Variables:</Typography.Text>
              {p.variables.map((v) => (
                <Tag key={v} color="blue">
                  {`{${v}}`}
                </Tag>
              ))}
            </Space>
          ) : null}
          <Input.TextArea
            value={text}
            onChange={(e) => {
              handleChange(p.key, e.target.value);
            }}
            rows={Math.min(20, Math.max(6, text.split("\n").length + 1))}
            disabled={isLoading}
          />
          <Button
            size="small"
            onClick={() => {
              handleChange(p.key, p.default_text);
            }}
            disabled={text === p.default_text}
          >
            Restore default text
          </Button>
        </Flex>
      ),
    };
  });

  return (
    <Flex vertical gap={16} style={{ width: "100%" }}>
      <Typography.Title level={3}>Model Prompts</Typography.Title>
      <Typography.Paragraph type="secondary" style={{ marginBottom: 0 }}>
        Edit the prompts used by the application when interacting with the
        model. Restart is required. Use &ldquo;Reset&rdquo; to revert a prompt to its default.
      </Typography.Paragraph>
      <Flex gap={8}>
        <Button
          type="primary"
          onClick={handleSave}
          loading={updatePromptsMutation.isPending}
          disabled={dirtyCount === 0}
        >
          Save Changes
          {dirtyCount > 0 ? ` (${String(dirtyCount)})` : ""}
        </Button>
        <Popconfirm title="Reset ALL prompts to their defaults?">
          <Button danger disabled={resetPromptsMutation.isPending}>
            Reset All
          </Button>
        </Popconfirm>
      </Flex>
      <Collapse items={items} defaultActiveKey={items.map((i) => i.key)} />
    </Flex>
  );
};

export default ModelPromptPage;
