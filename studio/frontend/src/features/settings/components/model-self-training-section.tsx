// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Switch } from "@/components/ui/switch";
import { useT } from "@/i18n";
import { useEffect, useState } from "react";
import {
  loadSelfTrainingSettings,
  type SelfTrainingSettings,
  updateSelfTrainingSettings,
} from "../api/self-training";
import { SettingsRow } from "./settings-row";
import { SettingsSection } from "./settings-section";

export function ModelSelfTrainingSection() {
  const t = useT();
  const [settings, setSettings] = useState<SelfTrainingSettings | null>(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    void loadSelfTrainingSettings()
      .then((next) => {
        if (!cancelled) setSettings(next);
      })
      .catch((cause: unknown) => {
        if (!cancelled) {
          setError(cause instanceof Error ? cause.message : String(cause));
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  async function save(enabled: boolean) {
    setSaving(true);
    setError(null);
    try {
      setSettings(await updateSelfTrainingSettings(enabled));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setSaving(false);
    }
  }

  return (
    <SettingsSection
      title={t("settings.general.modelSelfTraining.sectionTitle")}
      description={t("settings.general.modelSelfTraining.sectionDescription")}
    >
      <SettingsRow
        label={t("settings.general.modelSelfTraining.enableLabel")}
        description={t(
          "settings.general.modelSelfTraining.enableDescription",
        )}
        below={
          error ? (
            <span className="max-w-[calc(260px*var(--ui-space-scale,1))] text-right text-xs text-destructive">
              {error}
            </span>
          ) : null
        }
      >
        <Switch
          checked={settings?.enabled ?? false}
          disabled={!settings || saving}
          onCheckedChange={(enabled) => void save(enabled)}
        />
      </SettingsRow>
    </SettingsSection>
  );
}
