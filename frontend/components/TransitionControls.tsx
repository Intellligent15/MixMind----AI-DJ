"use client";

import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  api,
  isStatusError,
  type MixPlan,
  type Queue,
  type TransitionStyleInfo,
} from "@/lib/api";

/**
 * Per-transition controls for the mix: shows each pair's transition style,
 * the LLM's rationale, and which planner path produced it (LLM / repaired /
 * fallback — fallbacks are never silent anymore), with a style picker and a
 * Re-roll button. Re-rolling bumps a server-side nonce so the new plan is a
 * genuinely fresh sample, then re-renders just that pair and re-stitches.
 */

const SOURCE_LABELS: Record<string, { text: string; tone: string }> = {
  llm_v2: { text: "AI plan", tone: "text-emerald-400" },
  llm_v2_repaired: { text: "AI plan (repaired)", tone: "text-emerald-300" },
  llm_legacy: { text: "AI plan (legacy)", tone: "text-emerald-400" },
  llm_legacy_repaired: { text: "AI plan (legacy, repaired)", tone: "text-emerald-300" },
  style_default: { text: "default expansion", tone: "text-amber-400" },
  deterministic_fallback: { text: "deterministic fallback", tone: "text-amber-400" },
  deterministic: { text: "deterministic", tone: "text-zinc-400" },
  cached: { text: "cached", tone: "text-zinc-400" },
};

function prettyStyle(id: string | null): string {
  if (!id) return "—";
  return id.replaceAll("_", " ");
}

function pitchBadge(offset: number): string {
  const sign = offset > 0 ? "+" : "\u2212";
  return `${sign}${Math.abs(offset)} st`;
}

function TransitionRow({
  plan,
  fromTitle,
  toTitle,
  toPitchOffset,
  styles,
}: {
  plan: MixPlan;
  fromTitle: string;
  toTitle: string;
  toPitchOffset: number;
  styles: TransitionStyleInfo[];
}) {
  const queryClient = useQueryClient();
  const [pendingStyle, setPendingStyle] = useState<string>(
    plan.style_override ?? "auto"
  );

  const reroll = useMutation({
    mutationFn: () =>
      api.rerollMixPlan(plan.id, pendingStyle === "auto" ? undefined : pendingStyle),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["mix-plans", plan.queue_id] });
      // The Player's stitched-mix query lives under ["mix", queueId].
      // Invalidating it makes the Player refetch immediately, see the
      // reset (pending) status, and resume its 1s polling until the
      // re-stitched mix is ready — no page reload needed.
      queryClient.invalidateQueries({ queryKey: ["mix", plan.queue_id] });
    },
  });

  const source = plan.plan_source ? SOURCE_LABELS[plan.plan_source] : null;
  const busy = plan.status === "rendering" || reroll.isPending;
  const selectedInfo = styles.find((s) => s.id === pendingStyle);

  return (
    <li className="border border-zinc-800 rounded-md p-3 flex flex-col gap-2">
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-sm truncate">
          {fromTitle} <span className="opacity-50">→</span> {toTitle}
        </span>
        <span className="text-xs opacity-60 shrink-0">{plan.status}</span>
      </div>

      <div className="flex flex-wrap items-center gap-2 text-xs">
        <span className="px-2 py-0.5 rounded bg-zinc-800">
          {prettyStyle(plan.style)}
        </span>
        {source && <span className={source.tone}>{source.text}</span>}
        {plan.style_hint && !plan.style_override && (
          <span className="opacity-50">set plan suggested: {prettyStyle(plan.style_hint)}</span>
        )}
        {plan.style_override && (
          <span className="text-sky-400">pinned: {prettyStyle(plan.style_override)}</span>
        )}
        {toPitchOffset !== 0 && (
          <span
            className="px-2 py-0.5 rounded bg-violet-900/60 text-violet-200"
            title={`The incoming song plays ${pitchBadge(toPitchOffset)} for its entire duration so its key blends with the previous track \u2014 no mid-song key glide.`}
          >
            in: {pitchBadge(toPitchOffset)}
          </span>
        )}
      </div>

      {plan.rationale && (
        <p className="text-xs opacity-70 italic">“{plan.rationale}”</p>
      )}
      {plan.error_text && (
        <p className="text-xs text-red-400">{plan.error_text}</p>
      )}

      <div className="flex items-center gap-2">
        <select
          className="bg-zinc-900 border border-zinc-700 rounded px-2 py-1 text-xs"
          value={pendingStyle}
          onChange={(e) => setPendingStyle(e.target.value)}
          disabled={busy}
          title={selectedInfo?.description ?? "Let the AI pick the style"}
        >
          <option value="auto">style: auto</option>
          {styles.map((s) => (
            <option key={s.id} value={s.id} title={s.description}>
              {prettyStyle(s.id)}
            </option>
          ))}
        </select>
        <button
          onClick={() => reroll.mutate()}
          disabled={busy}
          className="px-3 py-1 text-xs rounded bg-zinc-100 text-zinc-900 disabled:opacity-40 hover:bg-white transition-colors"
        >
          {busy ? "Re-rolling…" : "Re-roll"}
        </button>
        {selectedInfo && (
          <span className="text-[10px] opacity-50 truncate">
            {selectedInfo.description}
          </span>
        )}
      </div>
    </li>
  );
}

export function TransitionControls() {
  const queueQuery = useQuery<Queue | null>({
    queryKey: ["queue", "current"],
    queryFn: async () => {
      try {
        return await api.getCurrentQueue();
      } catch (err) {
        if (isStatusError(err, 404)) return null;
        throw err;
      }
    },
  });
  const queue = queueQuery.data;

  const plansQuery = useQuery<MixPlan[]>({
    queryKey: ["mix-plans", queue?.id],
    queryFn: () => api.listMixPlansForQueue(queue!.id),
    enabled: !!queue?.id && queue.locked,
    refetchInterval: (q) => {
      const plans = q.state.data;
      if (!plans) return 2000;
      return plans.some((p) => p.status === "pending" || p.status === "rendering")
        ? 2000
        : false;
    },
  });

  const stylesQuery = useQuery<TransitionStyleInfo[]>({
    queryKey: ["transition-styles"],
    queryFn: api.listTransitionStyles,
    staleTime: Infinity,
  });

  if (!queue?.locked || !plansQuery.data?.length) return null;

  const titleBySong = new Map(
    queue.items.map((it) => [it.song.id, it.song.title])
  );
  const pitchBySong = new Map(
    queue.items.map((it) => [it.song.id, it.pitch_offset_semitones ?? 0])
  );

  return (
    <section className="flex flex-col gap-3">
      <h2 className="text-sm uppercase tracking-wide opacity-60">
        Transitions
      </h2>
      <ul className="flex flex-col gap-2">
        {plansQuery.data.map((plan) => (
          <TransitionRow
            key={plan.id}
            plan={plan}
            fromTitle={titleBySong.get(plan.from_song_id) ?? "?"}
            toTitle={titleBySong.get(plan.to_song_id) ?? "?"}
            toPitchOffset={pitchBySong.get(plan.to_song_id) ?? 0}
            styles={stylesQuery.data ?? []}
          />
        ))}
      </ul>
    </section>
  );
}
