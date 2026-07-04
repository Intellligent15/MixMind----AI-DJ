"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { useQueries, useQuery } from "@tanstack/react-query";
import WaveSurfer from "wavesurfer.js";
import {
  api,
  isStatusError,
  type Queue,
  type Song,
  type SongStatus,
} from "@/lib/api";
import { displayStatus } from "@/lib/song-status";

const PLAYABLE_STATUSES: ReadonlyArray<SongStatus> = [
  "downloaded",
  "analyzing",
  "analyzed",
  "separating",
  "transcribing",
  "ready",
];

function isPlayable(s: Song): boolean {
  return PLAYABLE_STATUSES.includes(s.status);
}

function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "0:00";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export function Player() {
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

  const items = queueQuery.data?.items ?? [];

  // Background polling of each queue song so a not-yet-downloaded next-up
  // becomes playable mid-set without a full queue refetch.
  const songQueries = useQueries({
    queries: items.map((item) => ({
      queryKey: ["song", item.song.id],
      queryFn: () => api.getSong(item.song.id),
      initialData: item.song,
      refetchInterval: (q: { state: { data?: Song } }) => {
        const s = q.state.data;
        if (!s) return 1000;
        if (s.status === "failed" || s.status === "ready") return false;
        return 1000;
      },
    })),
  });

  const songs: Song[] = useMemo(
    () => songQueries.map((q, i) => q.data ?? items[i].song),
    [songQueries, items]
  );

  const [currentIdx, setCurrentIdx] = useState(0);
  const [skipNotice, setSkipNotice] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [position, setPosition] = useState(0);
  const [duration, setDuration] = useState(0);
  const [autoplayBlocked, setAutoplayBlocked] = useState(false);

  const waveContainerRef = useRef<HTMLDivElement | null>(null);
  const wsRef = useRef<WaveSurfer | null>(null);

  const current = songs[currentIdx];
  const upcoming = songs[currentIdx + 1];

  const findNextPlayable = useCallback(
    (from: number): number | null => {
      for (let i = from; i < songs.length; i++) {
        if (isPlayable(songs[i])) return i;
      }
      return null;
    },
    [songs]
  );

  // Skip-forward fallback: if the current index isn't playable, advance to
  // the next playable index. Only applies once songs have loaded.
  useEffect(() => {
    if (!songs.length) return;
    const cur = songs[currentIdx];
    if (!cur) return;
    if (cur.status === "failed") {
      const next = findNextPlayable(currentIdx + 1);
      if (next != null && next !== currentIdx) {
        setSkipNotice(`Skipped: ${cur.title} (failed)`);
        setCurrentIdx(next);
      }
    }
  }, [songs, currentIdx, findNextPlayable]);

  const mixQuery = useQuery({
    queryKey: ["mix", queueQuery.data?.id],
    queryFn: () => queueQuery.data ? api.getQueueMix(queueQuery.data.id) : null,
    enabled: !!queueQuery.data?.id,
    refetchInterval: (q) => q.state.data?.status === "pending" || q.state.data?.status === "rendering" ? 1000 : false,
  });

  const mixData = mixQuery.data;

  const [playMode, setPlayMode] = useState<"mix" | "queue">("mix");
  const isMixReady = mixData?.status === "ready";
  const activeMode = isMixReady ? playMode : "queue";
  // Only meaningful in mix mode; null otherwise so status churn during a
  // re-render can't restart per-song playback.
  const mixAudioVersion =
    activeMode === "mix" ? mixData?.updated_at ?? null : null;

  const songById = useMemo(
    () => Object.fromEntries(songs.map((s) => [s.id, s])),
    [songs]
  );

  // Phase 10 transition indicator + upcoming preview. The stitched-mix
  // timeline maps output-seconds → which song leads and which transition is
  // active, so we derive both straight from the playhead `position`.
  const timeline = mixData?.timeline ?? null;
  const mixCurrentSong = useMemo(() => {
    if (activeMode !== "mix" || !timeline?.songs.length) return null;
    // Each song owns [start, end); the gaps BETWEEN those spans are the
    // transition regions (the timeline tracks transitions separately). During
    // a transition the playhead sits in that gap, so a strict [start, end)
    // containment finds nothing. Resolve "now playing" as the last song that
    // has started (start <= position): inside a song span that's the song
    // itself, and during a transition it's the outgoing song — which then
    // flips to the incoming song exactly when the transition ends. songs are
    // ordered by ascending start, so the last match wins.
    let lead = timeline.songs[0];
    for (const s of timeline.songs) {
      if (s.start <= position) lead = s;
      else break;
    }
    return lead;
  }, [activeMode, timeline, position]);
  const mixUpcomingSong = useMemo(() => {
    if (!timeline || !mixCurrentSong) return null;
    return (
      timeline.songs.find((s) => s.index === mixCurrentSong.index + 1) ?? null
    );
  }, [timeline, mixCurrentSong]);
  const activeTransition = useMemo(() => {
    if (activeMode !== "mix" || !timeline) return null;
    return (
      timeline.transitions.find(
        (t) => position >= t.start && position < t.end
      ) ?? null
    );
  }, [activeMode, timeline, position]);

  // Feedback window: the most recent transition that has started stays
  // votable for 30 s after it ends — and the FINAL transition stays
  // votable through the outro and after the mix finishes, since there's
  // nothing after it to confuse the vote with (and the mix often ends
  // before a 30 s window would).
  const feedbackTransition = useMemo(() => {
    if (activeMode !== "mix" || !timeline?.transitions.length) return null;
    let recent = null;
    for (const t of timeline.transitions) {
      if (t.start <= position) recent = t;
      else break;
    }
    if (!recent) return null;
    const last = timeline.transitions[timeline.transitions.length - 1];
    if (position < recent.end + 30 || recent.index === last.index) {
      return recent;
    }
    return null;
  }, [activeMode, timeline, position]);
  const [votedTransitions, setVotedTransitions] = useState<
    Record<number, "thumbs_up" | "thumbs_down">
  >({});
  const voteOnTransition = useCallback(
    (kind: "thumbs_up" | "thumbs_down") => {
      const qid = queueQuery.data?.id;
      const t = feedbackTransition;
      if (!qid || !t) return;
      setVotedTransitions((prev) => ({ ...prev, [t.index]: kind }));
      api
        .sendTransitionFeedback(qid, t.from_song_id, t.to_song_id, kind)
        .catch(() => {
          // Feedback is best-effort; never interrupt playback over it.
        });
    },
    [queueQuery.data?.id, feedbackTransition]
  );
  // Large forward seeks in mix mode count as skips — attributed
  // server-side to the transition the listener jumped away from.
  const lastPositionRef = useRef(0);
  useEffect(() => {
    lastPositionRef.current = position;
  }, [position]);

  // Live energy dial: bend the not-yet-played remainder of the set.
  const [dialNotice, setDialNotice] = useState<string | null>(null);
  const energyDial = useCallback(
    (direction: "up" | "hold" | "down") => {
      const qid = queueQuery.data?.id;
      if (!qid) return;
      api
        .setEnergyDial(qid, direction, lastPositionRef.current)
        .then((res) => {
          const n = res.affected_transitions.length;
          setDialNotice(
            n === 0
              ? "Nothing far enough ahead to change — try the skip button"
              : `Re-reading the room… ${n} transition${n === 1 ? "" : "s"} updating`
          );
          setTimeout(() => setDialNotice(null), 6000);
        })
        .catch((err) =>
          setDialNotice((err as Error).message ?? "energy dial failed")
        );
    },
    [queueQuery.data?.id]
  );

  // Instant skip: jump the playhead to just before the next transition
  // (and let the feedback loop know this stretch got skipped).
  const nextTransition = useMemo(() => {
    if (activeMode !== "mix" || !timeline) return null;
    return timeline.transitions.find((t) => t.start > position + 2) ?? null;
  }, [activeMode, timeline, position]);
  const skipToNextTransition = useCallback(() => {
    const ws = wsRef.current;
    const qid = queueQuery.data?.id;
    if (!ws || !nextTransition) return;
    if (qid) {
      api.sendPlaybackEvent(qid, "skip", lastPositionRef.current).catch(() => {});
    }
    ws.setTime(Math.max(0, nextTransition.start - 2));
  }, [nextTransition, queueQuery.data?.id]);

  // Hot-swap: when the mix re-renders mid-listen (energy dial / reroll),
  // the wavesurfer effect below recreates the player with the new file —
  // capture the playhead so "ready" can restore it. Content before the
  // first changed transition is time-identical, so the position maps 1:1.
  const resumePositionRef = useRef<number | null>(null);
  const prevMixKeyRef = useRef<string | null>(null);
  useEffect(() => {
    const key = activeMode === "mix" ? mixAudioVersion : null;
    if (
      prevMixKeyRef.current !== null &&
      key !== null &&
      key !== prevMixKeyRef.current
    ) {
      resumePositionRef.current = lastPositionRef.current;
    }
    prevMixKeyRef.current = key;
  }, [activeMode, mixAudioVersion]);

  // Auto-stitch backstop. The backend fires the eager stitch when the queue
  // finishes processing, so the mix row normally already exists by the time
  // the player mounts. This covers the gaps: no row at all, or a prior
  // render failed, while every song is ready. Fires at most once.
  const stitchRequestedRef = useRef(false);
  useEffect(() => {
    const qid = queueQuery.data?.id;
    if (!qid || songs.length < 2) return;
    if (!songs.every((s) => s.status === "ready")) return;
    if (stitchRequestedRef.current) return;
    if (mixData == null || mixData.status === "failed") {
      stitchRequestedRef.current = true;
      api
        .stitchQueue(qid)
        .then(() => mixQuery.refetch())
        .catch(() => {
          stitchRequestedRef.current = false;
        });
    }
  }, [songs, mixData, queueQuery.data?.id, mixQuery]);

  const handleRenderMix = async () => {
    if (!queueQuery.data) return;
    try {
      await api.stitchQueue(queueQuery.data.id);
      mixQuery.refetch();
    } catch (e) {
      console.error(e);
      alert("Failed to start mix render");
    }
  };

  const advance = useCallback(() => {
    const next = findNextPlayable(currentIdx + 1);
    if (next == null) {
      setIsPlaying(false);
      return;
    }
    if (next !== currentIdx + 1) {
      const skipped = songs
        .slice(currentIdx + 1, next)
        .map((s) => s.title)
        .join(", ");
      if (skipped) setSkipNotice(`Skipped: ${skipped} (not ready)`);
    } else {
      setSkipNotice(null);
    }
    setCurrentIdx(next);
  }, [currentIdx, findNextPlayable, songs]);

  const advanceRef = useRef(advance);
  useEffect(() => {
    advanceRef.current = advance;
  }, [advance]);

  const currentPlayable = current ? isPlayable(current) : false;

  const safeRegionsQuery = useQuery({
    queryKey: ["vocal_safe_regions", current?.id],
    queryFn: () => current ? api.getVocalSafeRegions(current.id) : null,
    // Only useful when we're actually rendering the per-song waveform.
    enabled: !!current?.id && activeMode === "queue",
    retry: false,
  });

  type RegionsPluginInstance = {
    clearRegions: () => void;
    addRegion: (opts: {
      start: number;
      end: number;
      content?: string;
      color?: string;
      drag?: boolean;
      resize?: boolean;
    }) => void;
  };
  const regionsPluginRef = useRef<RegionsPluginInstance | null>(null);

  // WaveSurfer setup
  useEffect(() => {
    if (!waveContainerRef.current) return;
    if (!mixData || mixData.status !== "ready") {
      if (!current || !currentPlayable) return;
    }

    setPosition(0);
    setDuration(0);

    // Version the mix URL with the render row's updated_at: a re-stitch
    // writes a new timestamp, producing a NEW url — so the browser can't
    // serve the previous mix from its HTTP cache, and this effect re-runs
    // (mixAudioVersion is a dependency) to load the fresh audio.
    const audioUrl = activeMode === "mix" && queueQuery.data
      ? `${api.queueMixAudioUrl(queueQuery.data.id)}?v=${encodeURIComponent(mixAudioVersion ?? "")}`
      : api.audioUrl(current!.id);

    let isCancelled = false;
    let localWs: WaveSurfer | null = null;

    import("wavesurfer.js/dist/plugins/regions.esm.js").then(({ default: RegionsPlugin }) => {
      if (isCancelled || !waveContainerRef.current) return;
      const regions = RegionsPlugin.create();
      regionsPluginRef.current = regions;
      
      const ws = WaveSurfer.create({
        container: waveContainerRef.current,
        waveColor: "#94a3b8",
        progressColor: "#0ea5e9",
        cursorColor: "#0ea5e9",
        height: 72,
        barWidth: 1,
        barGap: 1,
        barHeight: 0.6,
        url: audioUrl,
        plugins: [regions],
      });
      localWs = ws;
      wsRef.current = ws;

      ws.on("ready", () => {
        setDuration(ws.getDuration());
        // Restore the playhead after a mid-listen mix re-render (energy
        // dial / reroll hot-swap). One-shot.
        const resume = resumePositionRef.current;
        resumePositionRef.current = null;
        if (resume != null && resume > 1 && resume < ws.getDuration() - 1) {
          ws.setTime(resume);
        }
        ws.play().then(
          () => setAutoplayBlocked(false),
          () => setAutoplayBlocked(true),
        );
      });
      ws.on("play", () => {
        setIsPlaying(true);
        setAutoplayBlocked(false);
      });
      ws.on("pause", () => setIsPlaying(false));
      ws.on("timeupdate", (t: number) => setPosition(t));
      ws.on("interaction", (newTime: number) => {
        // A big forward jump in mix mode reads as "get me out of here".
        const from = lastPositionRef.current;
        const qid = queueQuery.data?.id;
        if (activeMode === "mix" && qid && newTime > from + 20) {
          api.sendPlaybackEvent(qid, "skip", from).catch(() => {});
        }
      });
      ws.on("finish", () => {
        if (activeMode !== "mix") advanceRef.current();
      });
    });

    return () => {
      isCancelled = true;
      if (localWs) {
        localWs.destroy();
      } else if (wsRef.current) {
        wsRef.current.destroy();
      }
      wsRef.current = null;
      regionsPluginRef.current = null;
    };
  }, [currentIdx, current?.id, currentPlayable, activeMode, queueQuery.data?.id, mixAudioVersion]);

  const plotRegions = useCallback(() => {
    const regionsPlugin = regionsPluginRef.current;
    if (!regionsPlugin) return;
    // Always clear first — covers the "flipped to mix mode" case
    // where stale per-song regions would otherwise persist.
    regionsPlugin.clearRegions();
    if (activeMode !== "queue") return;
    if (!safeRegionsQuery.data) return;
    safeRegionsQuery.data.regions.forEach((r) => {
      regionsPlugin.addRegion({
        start: r.start,
        end: r.end,
        // Drop the noisy "Safe" label — visual band is enough.
        content: "",
        color: "rgba(34, 197, 94, 0.2)",
        drag: false,
        resize: false,
      });
    });
  }, [safeRegionsQuery.data, activeMode]);

  useEffect(() => {
    plotRegions();
  }, [plotRegions]);

  const togglePlay = useCallback(() => {
    const ws = wsRef.current;
    if (!ws) return;
    if (ws.isPlaying()) ws.pause();
    else ws.play().catch(() => setAutoplayBlocked(true));
  }, []);

  if (queueQuery.isLoading) {
    return <p className="text-sm opacity-70">Loading…</p>;
  }
  if (!queueQuery.data || !queueQuery.data.locked || songs.length === 0) {
    return (
      <p className="text-sm opacity-70">
        No locked queue. Build and lock one on the{" "}
        <Link href="/" className="underline">
          home page
        </Link>
        .
      </p>
    );
  }
  
  

  if (!current && activeMode === "queue") {
    return (
      <p className="text-sm opacity-70">
        End of queue. Build a new one on the{" "}
        <Link href="/" className="underline">
          home page
        </Link>
        .
      </p>
    );
  }

  return (
    <div className="flex flex-col gap-6">
      {isMixReady && (
        <div className="flex bg-zinc-900 border rounded-lg p-1 self-start">
          <button
            onClick={() => setPlayMode("mix")}
            className={`px-4 py-1.5 text-sm font-medium rounded-md transition-colors ${
              playMode === "mix" ? "bg-blue-600 text-white" : "hover:bg-zinc-800 text-zinc-400"
            }`}
          >
            Continuous Mix
          </button>
          <button
            onClick={() => setPlayMode("queue")}
            className={`px-4 py-1.5 text-sm font-medium rounded-md transition-colors ${
              playMode === "queue" ? "bg-blue-600 text-white" : "hover:bg-zinc-800 text-zinc-400"
            }`}
          >
            Queue Mode
          </button>
        </div>
      )}
      
      {skipNotice && activeMode === "queue" && (
        <div className="border border-yellow-500/40 rounded p-2 text-sm bg-yellow-500/10">
          {skipNotice}
        </div>
      )}
      
      {!isMixReady && (
        <section className="border rounded p-4 flex gap-4 items-center bg-zinc-900">
          <div className="flex-1">
            <h3 className="font-semibold text-lg">Continuous DJ Mix</h3>
            <p className="text-sm opacity-70">Render all transitions into a single continuous mix.</p>
            {mixData && (mixData.status === "pending" || mixData.status === "rendering") && (
              <p className="text-sm text-yellow-500 mt-2">Rendering mix...</p>
            )}
            {mixData && mixData.status === "failed" && (
              <p className="text-sm text-red-500 mt-2">Render failed: {mixData.error_text}</p>
            )}
          </div>
          <button
            onClick={handleRenderMix}
            disabled={mixData?.status === "pending" || mixData?.status === "rendering"}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded font-medium disabled:opacity-50"
          >
            Render Full Mix
          </button>
        </section>
      )}

      {activeMode === "mix" ? (
        <section className="border rounded p-4 flex gap-4">
          {mixCurrentSong &&
            songById[mixCurrentSong.song_id]?.thumbnail_url && (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={songById[mixCurrentSong.song_id].thumbnail_url!}
                alt=""
                className="w-32 h-20 object-cover rounded"
              />
            )}
          <div className="flex-1 min-w-0 flex flex-col gap-1">
            <p className="text-xs opacity-60">
              Now playing · continuous mix
              {mixCurrentSong
                ? ` · ${mixCurrentSong.index + 1}/${songs.length}`
                : ""}
            </p>
            <p className="font-semibold truncate text-lg">
              {mixCurrentSong?.title ?? "Full DJ Mix"}
            </p>
            <p className="text-sm opacity-70 truncate">
              {mixCurrentSong?.artist ?? `${songs.length} tracks`}
            </p>
            <a
              href={api.queueMixAudioUrl(queueQuery.data.id)}
              download="mix.m4a"
              className="text-blue-400 text-sm hover:underline mt-2 self-start"
            >
              Download Audio
            </a>
          </div>
        </section>
      ) : (
        <section className="border rounded p-4 flex gap-4">
          {current.thumbnail_url && (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={current.thumbnail_url}
              alt=""
              className="w-32 h-20 object-cover rounded"
            />
          )}
          <div className="flex-1 min-w-0 flex flex-col gap-1">
            <p className="text-xs opacity-60">Now playing · {currentIdx + 1}/{songs.length}</p>
            <p className="font-semibold truncate text-lg">{current.title}</p>
            <p className="text-sm opacity-70 truncate">{current.artist ?? "—"}</p>
            <p className="text-xs opacity-60 mt-1">status: {displayStatus(current)}</p>
          </div>
        </section>
      )}

      {dialNotice && (
        <p className="text-xs opacity-70 border border-amber-500/40 bg-amber-500/10 rounded px-3 py-2">
          {dialNotice}
        </p>
      )}

      {activeMode === "mix" &&
        (timeline?.host ?? []).some(
          (h) => position >= h.start && position < h.end + 1
        ) && (
          <p className="text-xs border border-purple-500/40 bg-purple-500/10 rounded px-3 py-2">
            Host:{" "}
            {
              timeline!.host!.find(
                (h) => position >= h.start && position < h.end + 1
              )!.text
            }
          </p>
        )}

      {activeMode === "mix" && activeTransition && (
        <section className="border border-blue-500/40 bg-blue-500/10 rounded p-3 flex flex-col gap-1">
          <div className="flex items-center justify-between">
            <p className="text-xs opacity-70">Transition in progress</p>
            <div className="flex items-center gap-1">
              {votedTransitions[activeTransition.index] ? (
                <span className="text-xs opacity-60">
                  noted (
                  {votedTransitions[activeTransition.index] === "thumbs_up"
                    ? "good"
                    : "bad"}
                  )
                </span>
              ) : (
                <>
                  <button
                    type="button"
                    onClick={() => voteOnTransition("thumbs_up")}
                    title="This transition works"
                    className="text-xs border rounded px-2 py-0.5 hover:bg-black/5 dark:hover:bg-white/10"
                  >
                    Good
                  </button>
                  <button
                    type="button"
                    onClick={() => voteOnTransition("thumbs_down")}
                    title="Not feeling this one"
                    className="text-xs border rounded px-2 py-0.5 hover:bg-black/5 dark:hover:bg-white/10"
                  >
                    Bad
                  </button>
                </>
              )}
            </div>
          </div>
          <p className="font-medium">{activeTransition.label}</p>
          {activeTransition.stems.length > 0 && (
            <p className="text-xs opacity-80">
              {activeTransition.stems
                .map((s) => `${s.stem} ${s.from}→${s.to}`)
                .join(" · ")}
            </p>
          )}
          {activeTransition.effects.length > 0 && (
            <p className="text-xs opacity-60">
              + {activeTransition.effects.join(", ")}
            </p>
          )}
          <p className="text-[11px] opacity-50">
            A = {songById[activeTransition.from_song_id]?.title ?? "outgoing"} ·
            B = {songById[activeTransition.to_song_id]?.title ?? "incoming"}
          </p>
        </section>
      )}

      {activeMode === "mix" && !activeTransition && feedbackTransition && (
        <section className="border rounded px-3 py-2 flex items-center justify-between gap-2">
          <p className="text-xs opacity-70 truncate">
            That {feedbackTransition.label.toLowerCase()} transition
            {songById[feedbackTransition.to_song_id]?.title
              ? ` into ${songById[feedbackTransition.to_song_id]!.title}`
              : ""}{" "}
            — how was it?
          </p>
          {votedTransitions[feedbackTransition.index] ? (
            <span className="text-xs opacity-60 shrink-0">
              noted (
              {votedTransitions[feedbackTransition.index] === "thumbs_up"
                ? "good"
                : "bad"}
              )
            </span>
          ) : (
            <div className="flex items-center gap-1 shrink-0">
              <button
                type="button"
                onClick={() => voteOnTransition("thumbs_up")}
                className="text-xs border rounded px-2 py-0.5 hover:bg-black/5 dark:hover:bg-white/10"
              >
                Good
              </button>
              <button
                type="button"
                onClick={() => voteOnTransition("thumbs_down")}
                className="text-xs border rounded px-2 py-0.5 hover:bg-black/5 dark:hover:bg-white/10"
              >
                Bad
              </button>
            </div>
          )}
        </section>
      )}

      <div className="relative">
        <div ref={waveContainerRef} className="border rounded p-2" />
        {autoplayBlocked && !isPlaying && (
          <button
            type="button"
            onClick={togglePlay}
            className="absolute inset-0 flex items-center justify-center rounded bg-black/40 text-white text-sm font-medium"
          >
            Click to start playback
          </button>
        )}
      </div>

      <section className="flex items-center gap-4">
        <button
          type="button"
          onClick={togglePlay}
          className="border rounded px-4 py-2 hover:bg-black/5 dark:hover:bg-white/10"
        >
          {isPlaying ? "Pause" : "Play"}
        </button>
        {activeMode === "queue" && (
          <button
            type="button"
            onClick={advance}
            className="border rounded px-4 py-2 hover:bg-black/5 dark:hover:bg-white/10"
          >
            Next
          </button>
        )}
        {activeMode === "mix" && (
          <>
            <button
              type="button"
              onClick={skipToNextTransition}
              disabled={!nextTransition}
              title="Jump to just before the next transition"
              className="border rounded px-4 py-2 hover:bg-black/5 dark:hover:bg-white/10 disabled:opacity-40"
            >
              Next transition
            </button>
            <div
              className="flex items-center gap-1 border rounded px-2 py-1"
              title="Bend the rest of the set (takes effect ~90s ahead)"
            >
              <span className="text-xs opacity-60 mr-1">Energy</span>
              <button
                type="button"
                onClick={() => energyDial("down")}
                title="Cool the rest of the set down"
                className="text-xs rounded px-2 py-1 hover:bg-black/5 dark:hover:bg-white/10"
              >
                Down
              </button>
              <button
                type="button"
                onClick={() => energyDial("hold")}
                className="text-xs rounded px-2 py-1 opacity-70 hover:opacity-100 hover:bg-black/5 dark:hover:bg-white/10"
              >
                hold
              </button>
              <button
                type="button"
                onClick={() => energyDial("up")}
                title="Raise the energy of the rest of the set"
                className="text-xs rounded px-2 py-1 hover:bg-black/5 dark:hover:bg-white/10"
              >
                Up
              </button>
            </div>
          </>
        )}
        <span className="text-sm tabular-nums opacity-70">
          {formatTime(position)} / {formatTime(duration)}
        </span>
      </section>

      {upcoming && activeMode === "queue" && (
        <section className="border rounded p-3 flex items-center gap-3 opacity-80">
          <span className="text-xs opacity-60">Next up</span>
          {upcoming.thumbnail_url && (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={upcoming.thumbnail_url}
              alt=""
              className="w-16 h-10 object-cover rounded"
            />
          )}
          <div className="flex-1 min-w-0">
            <p className="font-medium truncate">{upcoming.title}</p>
            <p className="text-xs opacity-70 truncate">
              {upcoming.artist ?? "—"} · {displayStatus(upcoming)}
            </p>
          </div>
        </section>
      )}

      {activeMode === "mix" && mixUpcomingSong && (
        <section className="border rounded p-3 flex items-center gap-3 opacity-80">
          <span className="text-xs opacity-60">Next up</span>
          {songById[mixUpcomingSong.song_id]?.thumbnail_url && (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={songById[mixUpcomingSong.song_id].thumbnail_url!}
              alt=""
              className="w-16 h-10 object-cover rounded"
            />
          )}
          <div className="flex-1 min-w-0">
            <p className="font-medium truncate">{mixUpcomingSong.title}</p>
            <p className="text-xs opacity-70 truncate">
              {mixUpcomingSong.artist ?? "—"}
            </p>
          </div>
        </section>
      )}

      <section className="border-t pt-4">
        <p className="text-xs opacity-60 mb-2">Queue</p>
        <ul className="flex flex-col gap-1 text-sm">
          {songs.map((s, i) => (
            <li
              key={s.id + ":" + i}
              className={
                "flex items-center gap-2 py-1 " +
                (i === currentIdx && activeMode === "queue" ? "font-semibold" : "opacity-70")
              }
            >
              <span className="w-6 tabular-nums">{i + 1}</span>
              <span className="flex-1 truncate">{s.title}</span>
              <span className="text-xs opacity-60">{displayStatus(s)}</span>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
