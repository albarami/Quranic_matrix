"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState, type ReactNode } from "react";
import { LanguageProvider } from "./contexts/LanguageContext";

interface ProvidersProps {
  children: ReactNode;
}

export function Providers({ children }: ProvidersProps) {
  // Create a new QueryClient instance for each session
  // This prevents shared state between different users/requests
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            // Retry failed requests twice before showing error
            retry: 2,
            // Don't refetch when window regains focus (less network traffic)
            refetchOnWindowFocus: false,
            // Keep data fresh for 2 minutes by default
            staleTime: 2 * 60 * 1000,
            // Cache data for 10 minutes even when not in use
            gcTime: 10 * 60 * 1000,
          },
          mutations: {
            // Retry mutations once on failure
            retry: 1,
          },
        },
      })
  );

  return (
    <QueryClientProvider client={queryClient}>
      <LanguageProvider>{children}</LanguageProvider>
    </QueryClientProvider>
  );
}
