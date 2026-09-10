/*******************************************************************************
 * CLOUDERA APPLIED MACHINE LEARNING PROTOTYPE (AMP)
 * (C) Cloudera, Inc. 2024
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

import React, { useRef, useState } from "react";
import {
  CommentOutlined,
  DatabaseOutlined,
  LineChartOutlined,
  SettingOutlined,
} from "@ant-design/icons";
import {
  Flex,
  Image,
  Layout,
  Menu,
  MenuProps,
  Typography,
} from "antd";
import { useMatchRoute, useNavigate } from "@tanstack/react-router";
import Images from "src/components/images/Images.ts";
import "./style.css";
import AmpUpdateBanner from "src/components/AmpUpdate/AmpUpdateBanner.tsx";
import { useGetPollingAmpConfig } from "src/api/ampMetadataApi.ts";
import { getItem } from "./TopNav";
import { useSuspenseQuery } from "@tanstack/react-query";

const { Sider } = Layout;

type MenuItem = Required<MenuProps>["items"][number];

const Sidebar: React.FC = () => {
  const [collapsed, setCollapsed] = useState(false);
  const matchRoute = useMatchRoute();
  const navigate = useNavigate();
  const ref = useRef<HTMLDivElement>(null);
  const { data: config } = useSuspenseQuery(useGetPollingAmpConfig());

  const navToRagApp = () => {
    navigate({ to: "/chats" }).catch(() => null);
  };

  const navToData = () => {
    navigate({ to: "/data" }).catch(() => null);
  };

  const navToAnalytics = () => {
    navigate({ to: "/analytics" }).catch(() => null);
  };

  const navToSettings = () => {
    navigate({ to: "/settings" }).catch(() => null);
  };

  const isValidConfig = Boolean(config && !config.is_valid_config);
  const enableFullUsage = Boolean(!config?.is_valid_config);

  const baseItems: MenuItem[] = [
    getItem({
      label: <div data-testid="rag-apps-nav">Chats</div>,
      key: "chat",
      disabled: isValidConfig,
      onClick: navToRagApp,
      icon: <CommentOutlined />,
      config,
    }),
    getItem({
      label: <div data-testid="data-management-nav">Knowledge Bases</div>,
      key: "data",
      disabled: enableFullUsage,
      onClick: navToData,
      icon: <DatabaseOutlined />,
      config,
    }),
  ];

  const analyticsItem = getItem({
    label: <div data-testid="analytics-nav">Analytics</div>,
    key: "analytics",
    disabled: enableFullUsage,
    onClick: navToAnalytics,
    icon: <LineChartOutlined />,
    config,
  });

  const settingsItem = getItem({
    label: <div data-testid="settings-nav">Settings</div>,
    key: "settings",
    disabled: enableFullUsage,
    onClick: navToSettings,
    icon: <SettingOutlined />,
    config,
  });

  const items = [...baseItems, analyticsItem];

  if (config) {
    items.push(settingsItem);
  }

  function chooseRoute() {
    if (matchRoute({ to: "/data", fuzzy: true })) {
      return ["data"];
    } else if (matchRoute({ to: "/chats", fuzzy: true })) {
      return ["chat"];
    } else if (matchRoute({ to: "/analytics", fuzzy: true })) {
      return ["analytics"];
    } else if (matchRoute({ to: "/projects", fuzzy: true })) {
      return ["projects"];
    } else if (matchRoute({ to: "/settings", fuzzy: true })) {
      return ["settings"];
    } else {
      return ["chat"];
    }
  }

  return (
    <Sider
      collapsible
      collapsed={collapsed}
      onCollapse={(value) => {
        setCollapsed(value);
      }}
      style={{
        transition: "none",
        height: "100vh",
        top: 0,
        position: "sticky",
      }}
      width={250}
      ref={ref}
    >
      {/* Centered Brand Header */}
      <Flex
        vertical
        align="center"
        justify="center"
        style={{ padding: "20px 16px", width: "100%" }}
      >
        <Image
          src={Images.BniIcon}
          preview={false}
          height={32}
          style={{ objectFit: "contain" }}
        />
        {!collapsed && (
          <Typography.Text
            style={{
              color: "rgba(255, 255, 255, 0.85)",
              fontWeight: 600,
              fontSize: 13,
              lineHeight: 1.2,
              whiteSpace: "nowrap",
              marginTop: 6,
              textAlign: "center",
            }}
          >
            Document Intelligence
          </Typography.Text>
        )}
      </Flex>

      <Flex vertical justify="space-between" style={{ height: "85%" }}>
        <Menu selectedKeys={chooseRoute()} mode="inline" items={items} />
        <AmpUpdateBanner isCollapsed={collapsed} />
      </Flex>
    </Sider>
  );
};

export default Sidebar;