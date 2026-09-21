/*
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
 * Absent a written agreement with Cloudera, Inc. ("Cloudera") to the
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

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ReactNode } from "react";
import { PageHeaderUserGreeting, UserGreeting } from "./UserGreeting";
import { getAmpIsComposableQueryOptions } from "src/api/ampMetadataApi.ts";

let mockUsername: string | undefined = "jdoe";
let mockCurrentUserError = false;
let mockIsComposable = false;

vi.mock("src/api/userApi.ts", () => ({
  getCurrentUserQueryOptions: {
    queryKey: ["getCurrentUser"],
    staleTime: Infinity,
    queryFn: () => {
      if (mockCurrentUserError) {
        return Promise.reject(new Error("failed to load user"));
      }
      return Promise.resolve({ username: mockUsername });
    },
  },
}));

vi.mock("src/api/ampMetadataApi.ts", async () => {
  const actual = await vi.importActual<typeof import("src/api/ampMetadataApi.ts")>(
    "src/api/ampMetadataApi.ts",
  );
  return {
    ...actual,
    getAmpIsComposableQueryOptions: {
      queryKey: ["getAmpIsComposable"],
      queryFn: () => Promise.resolve(mockIsComposable),
    },
  };
});

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  queryClient.setQueryData(getAmpIsComposableQueryOptions.queryKey, mockIsComposable);
  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
  return { wrapper, queryClient };
};

describe("UserGreeting", () => {
  afterEach(() => {
    cleanup();
    mockUsername = "jdoe";
    mockCurrentUserError = false;
    mockIsComposable = false;
  });

  it("renders a welcome message for the current user", async () => {
    const { wrapper } = createWrapper();
    render(<UserGreeting />, { wrapper });

    expect(await screen.findByTestId("welcome-user")).toHaveTextContent(
      "Welcome, jdoe",
    );
  });

  it("renders nothing when no user could be identified", async () => {
    mockUsername = "unknown";
    const { wrapper } = createWrapper();
    render(<UserGreeting />, { wrapper });

    await waitFor(() => {
      expect(screen.queryByTestId("welcome-user")).not.toBeInTheDocument();
    });
  });

  it("renders nothing when the request fails", async () => {
    mockCurrentUserError = true;
    const { wrapper } = createWrapper();
    render(<UserGreeting />, { wrapper });

    await waitFor(() => {
      expect(screen.queryByTestId("welcome-user")).not.toBeInTheDocument();
    });
  });
});

describe("PageHeaderUserGreeting", () => {
  afterEach(() => {
    cleanup();
    mockIsComposable = false;
  });

  it("renders the greeting in studio mode", async () => {
    mockIsComposable = false;
    const { wrapper } = createWrapper();
    render(<PageHeaderUserGreeting />, { wrapper });

    expect(await screen.findByTestId("welcome-user")).toHaveTextContent(
      "Welcome, jdoe",
    );
  });

  it("renders nothing in composable mode (the top nav already shows it)", async () => {
    mockIsComposable = true;
    const { wrapper } = createWrapper();
    render(<PageHeaderUserGreeting />, { wrapper });

    await waitFor(() => {
      expect(screen.queryByTestId("welcome-user")).not.toBeInTheDocument();
    });
  });
});
