// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";

export interface ComputeTargetSelectionState {
  selectedId: string | null;
  setSelected: (id: string | null) => void;
}

/**
 * The target the user picked. The picker stays prop-driven; whoever mounts it
 * reads and writes this store.
 */
export const useComputeTargetStore = create<ComputeTargetSelectionState>(
  (set) => ({
    selectedId: null,
    setSelected: (selectedId) => set({ selectedId }),
  }),
);
