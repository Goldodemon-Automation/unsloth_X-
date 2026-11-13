// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import type { ReactElement } from "react";
import type { TrainSubTab } from "./use-studio-navigation";

export function TrainSubNav({
  value,
  trainingRunActive,
  showTrainingView,
}: {
  value: TrainSubTab;
  trainingRunActive: boolean;
  showTrainingView: boolean;
}): ReactElement {
  const t = useT();
  const items: ReadonlyArray<{
    value: TrainSubTab;
    label: string;
    disabled: boolean;
  }> = [
    {
      value: "configure",
      label: t("studio.tabs.configure"),
      disabled: trainingRunActive,
    },
    {
      value: "current-run",
      label: t("studio.tabs.currentRun"),
      disabled: !showTrainingView,
    },
    { value: "history", label: t("studio.tabs.history"), disabled: false },
    { value: "compare", label: t("studio.tabs.compare"), disabled: false },
  ];
  return (
    <TabsList
      unstyled={true}
      data-tour="studio-subnav"
      // A macOS segmented control: one recessed track, the active segment a raised pill.
      // The underline rail it replaces had no visible container, so on a page with four
      // sections the tabs read as stray words rather than as one control.
      className="train-segmented flex min-w-0 flex-wrap items-center justify-start text-ui-13 tracking-nav"
    >
      {items.map((item) => {
        const active = value === item.value;
        return (
          <TabsTrigger
            key={item.value}
            value={item.value}
            disabled={item.disabled}
            indicatorClassName="hidden"
            className={cn(
              "train-segment relative h-8 flex-none select-none rounded-full border-0 px-3.5 py-0 text-ui-13 transition-colors disabled:cursor-not-allowed disabled:opacity-40",
              active
                ? "train-segment-active font-semibold text-background"
                : "text-muted-foreground hover:text-foreground",
            )}
          >
            {item.label}
          </TabsTrigger>
        );
      })}
    </TabsList>
  );
}
