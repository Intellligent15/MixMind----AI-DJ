"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import {
  DndContext,
  PointerSensor,
  closestCenter,
  useSensor,
  useSensors,
  type DragEndEvent,
} from "@dnd-kit/core";
import {
  SortableContext,
  arrayMove,
  verticalListSortingStrategy,
} from "@dnd-kit/sortable";
import {
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";
import {
  api,
  isStatusError,
  type OrderSuggestion,
  type Queue,
  type QueueContextUpdate,
  type SearchResult,
} from "@/lib/api";
import { QueueItemRow } from "./QueueItemRow";

function GigContext({ queue }: { queue: Queue }) {
  const qc = useQueryClient();
  const options = useQuery({
    queryKey: ["queue-context-options"],
    queryFn: api.getContextOptions,
    staleTime: Infinity,
  });
  const [note, setNote] = useState(queue.vibe_note ?? "");
  const update = useMutation({
    mutationFn: (ctx: QueueContextUpdate) =>
      api.updateQueueContext(queue.id, ctx),
    onSuccess: (q) => qc.setQueryData(["queue", "current"], q),
  });

  if (!options.data) return null;
  return (
    <div className="flex flex-col gap-2 border rounded p-3 mb-4">
      <p className="text-xs font-semibold opacity-70">The gig</p>
      <div className="flex flex-wrap gap-1.5">
        {options.data.occasions.map((o) => (
          <button
            key={o.id}
            type="button"
            onClick={() =>
              update.mutate({
                occasion: queue.occasion === o.id ? null : o.id,
              })
            }
            className={
              "text-xs border rounded-full px-2.5 py-1 " +
              (queue.occasion === o.id
                ? "bg-emerald-500/30 border-emerald-500/50"
                : "hover:bg-black/5 dark:hover:bg-white/10")
            }
          >
            {o.label}
          </button>
        ))}
      </div>
      <div className="flex gap-2 items-center">
        <select
          value={queue.arc_template ?? ""}
          onChange={(e) =>
            update.mutate({ arc_template: e.target.value || null })
          }
          className="text-xs border rounded px-2 py-1 bg-transparent"
        >
          <option value="">Arc: auto</option>
          {options.data.arcs.map((a) => (
            <option key={a.id} value={a.id} title={a.description}>
              Arc: {a.label}
            </option>
          ))}
        </select>
        <label className="text-xs flex items-center gap-1.5 cursor-pointer">
          <input
            type="checkbox"
            checked={queue.tease_hooks}
            onChange={(e) => update.mutate({ tease_hooks: e.target.checked })}
          />
          Tease hooks
        </label>
      </div>
      <div className="flex gap-2 items-center">
        <select
          value={queue.host_frequency ?? ""}
          onChange={(e) =>
            update.mutate({ host_frequency: e.target.value || null })
          }
          title="How often the host voice talks over the mix"
          className="text-xs border rounded px-2 py-1 bg-transparent"
        >
          <option value="">Host: default</option>
          {options.data.host_frequencies.map((f) => (
            <option key={f} value={f}>
              Host: {f.replace("_", " ")}
            </option>
          ))}
        </select>
        <select
          value={queue.host_persona ?? ""}
          onChange={(e) =>
            update.mutate({ host_persona: e.target.value || null })
          }
          title="The host's voice personality"
          className="text-xs border rounded px-2 py-1 bg-transparent"
        >
          <option value="">Voice: auto</option>
          {options.data.host_personas.map((p) => (
            <option key={p} value={p}>
              Voice: {p}
            </option>
          ))}
        </select>
      </div>
      <input
        value={note}
        onChange={(e) => setNote(e.target.value)}
        onBlur={() => {
          if (note !== (queue.vibe_note ?? "")) {
            update.mutate({ vibe_note: note || null });
          }
        }}
        maxLength={300}
        placeholder="Vibe note for the DJ (optional) — e.g. '90s hip-hop birthday, keep it fun'"
        className="text-xs border rounded px-2 py-1.5 bg-transparent"
      />
    </div>
  );
}

const QUEUE_CAP = 20;
const NON_TERMINAL: ReadonlySet<string> = new Set([
  "pending",
  "downloading",
  "downloaded",
  "analyzing",
  "separating",
  "transcribing",
]);

function useCurrentQueue() {
  const qc = useQueryClient();
  const query = useQuery<Queue | null>({
    queryKey: ["queue", "current"],
    queryFn: async () => {
      try {
        return await api.getCurrentQueue();
      } catch (err) {
        if (isStatusError(err, 404)) return null;
        throw err;
      }
    },
    // Poll while the queue is unlocked and any song is mid-pipeline, so
    // status pills update without the user having to interact.
    refetchInterval: (q) => {
      const data = q.state.data as Queue | null | undefined;
      if (data === undefined) return 1000;
      if (data === null) return false;
      if (data.locked) return false;
      const anyMoving = data.items.some((i) => NON_TERMINAL.has(i.song.status));
      return anyMoving ? 1500 : false;
    },
  });

  // Lazy-create on first load if none exists.
  useEffect(() => {
    if (query.isSuccess && query.data === null) {
      api.createQueue().then((q) => {
        qc.setQueryData(["queue", "current"], q);
      });
    }
  }, [query.isSuccess, query.data, qc]);

  return query;
}

function QueueSearch({
  onAdd,
  disabled,
}: {
  onAdd: (r: SearchResult) => Promise<void>;
  disabled: boolean;
}) {
  const [text, setText] = useState("");
  const [submitted, setSubmitted] = useState("");
  const results = useQuery({
    queryKey: ["search", submitted],
    queryFn: () => api.search(submitted, 10),
    enabled: submitted.length > 0,
  });

  return (
    <section className="flex flex-col gap-3">
      <form
        onSubmit={(e) => {
          e.preventDefault();
          setSubmitted(text.trim());
        }}
        className="flex gap-2"
      >
        <input
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Search YouTube..."
          className="flex-1 border rounded px-3 py-2 bg-transparent"
        />
        <button
          type="submit"
          className="border rounded px-4 py-2 hover:bg-black/5 dark:hover:bg-white/10"
        >
          Search
        </button>
      </form>
      {results.isFetching && (
        <p className="text-sm opacity-70">Searching…</p>
      )}
      {results.error && (
        <p className="text-sm text-red-600">
          {(results.error as Error).message}
        </p>
      )}
      <ul className="flex flex-col gap-2">
        {results.data?.map((r) => (
          <li
            key={r.youtube_video_id}
            className="flex items-center gap-3 border rounded p-2"
          >
            {r.thumbnail_url && (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={r.thumbnail_url}
                alt=""
                className="w-20 h-12 object-cover rounded"
              />
            )}
            <div className="flex-1 min-w-0">
              <p className="font-medium truncate">{r.title}</p>
              <p className="text-xs opacity-70 truncate">{r.artist ?? "—"}</p>
            </div>
            <button
              type="button"
              onClick={() => onAdd(r)}
              disabled={disabled}
              className="text-sm border rounded px-3 py-1 hover:bg-black/5 dark:hover:bg-white/10 disabled:opacity-50"
            >
              Add
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}

export function QueueBuilder() {
  const router = useRouter();
  const qc = useQueryClient();
  const queue = useCurrentQueue();
  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 6 } })
  );

  // Local optimistic ordering during a drag, applied on top of the server data.
  const [localOrder, setLocalOrder] = useState<string[] | null>(null);
  const itemsInOrder = useMemo(() => {
    const items = queue.data?.items ?? [];
    if (!localOrder) return items;
    const byId = new Map(items.map((i) => [i.id, i]));
    const ordered = localOrder
      .map((id) => byId.get(id))
      .filter((i): i is NonNullable<typeof i> => Boolean(i));
    // Server items not in localOrder (race) appended at the end.
    for (const i of items) if (!localOrder.includes(i.id)) ordered.push(i);
    return ordered;
  }, [queue.data, localOrder]);

  const addToQueue = useMutation({
    mutationFn: async (r: SearchResult) => {
      if (!queue.data) throw new Error("queue not ready");
      const song = await api.createSong(r);
      return api.addToQueue(queue.data.id, song.id);
    },
    onSuccess: (q) => {
      qc.setQueryData(["queue", "current"], q);
      qc.invalidateQueries({ queryKey: ["songs"] });
    },
  });

  const remove = useMutation({
    mutationFn: (itemId: string) => {
      if (!queue.data) throw new Error("queue not ready");
      return api.removeFromQueue(queue.data.id, itemId);
    },
    onSuccess: (q) => {
      setLocalOrder(null);
      qc.setQueryData(["queue", "current"], q);
    },
  });

  const reorder = useMutation({
    mutationFn: (orderedIds: string[]) => {
      if (!queue.data) throw new Error("queue not ready");
      return api.reorderQueue(queue.data.id, orderedIds);
    },
    onSuccess: (q) => {
      setLocalOrder(null);
      qc.setQueryData(["queue", "current"], q);
    },
    onError: () => {
      setLocalOrder(null);
    },
  });

  const lock = useMutation({
    mutationFn: () => {
      if (!queue.data) throw new Error("queue not ready");
      return api.lockQueue(queue.data.id);
    },
    onSuccess: (q) => {
      qc.setQueryData(["queue", "current"], q);
      router.push("/processing");
    },
  });

  function onDragEnd(event: DragEndEvent) {
    const { active, over } = event;
    if (!over || active.id === over.id) return;
    const current = itemsInOrder.map((i) => i.id);
    const oldIndex = current.indexOf(String(active.id));
    const newIndex = current.indexOf(String(over.id));
    if (oldIndex < 0 || newIndex < 0) return;
    const next = arrayMove(current, oldIndex, newIndex);
    setLocalOrder(next);
    reorder.mutate(next);
  }

  const startNew = useMutation({
    mutationFn: () => api.createQueue(),
    onSuccess: (q) => qc.setQueryData(["queue", "current"], q),
  });

  // Auto-ordering: fetch a proposal, show the diff, apply via the normal
  // reorder endpoint. First track stays pinned (server default).
  const [suggestion, setSuggestion] = useState<OrderSuggestion | null>(null);
  const suggest = useMutation({
    mutationFn: () => {
      if (!queue.data) throw new Error("queue not ready");
      return api.suggestOrder(queue.data.id);
    },
    onSuccess: (s) => setSuggestion(s),
  });
  const applySuggestion = () => {
    if (!suggestion) return;
    setLocalOrder(suggestion.ordered_item_ids);
    reorder.mutate(suggestion.ordered_item_ids);
    setSuggestion(null);
  };

  if (queue.isLoading || !queue.data) {
    return <p className="text-sm opacity-70">Loading queue…</p>;
  }

  // The current queue is locked — show the resume CTAs instead of a builder
  // that would 409 on every mutation. User can also start a fresh queue.
  if (queue.data.locked) {
    return (
      <div className="flex flex-col gap-4">
        <p className="text-sm opacity-70">
          The current queue is locked ({queue.data.items.length} song
          {queue.data.items.length === 1 ? "" : "s"}). Pick up where you left
          off or start a new one.
        </p>
        <div className="flex flex-wrap gap-3">
          <a
            href="/processing"
            className="border rounded px-4 py-2 text-sm hover:bg-black/5 dark:hover:bg-white/10"
          >
            Open processing view
          </a>
          <a
            href="/player"
            className="border rounded px-4 py-2 text-sm hover:bg-black/5 dark:hover:bg-white/10"
          >
            Open player
          </a>
          <button
            type="button"
            onClick={() => startNew.mutate()}
            disabled={startNew.isPending}
            className="border rounded px-4 py-2 text-sm hover:bg-black/5 dark:hover:bg-white/10 disabled:opacity-50"
          >
            {startNew.isPending ? "Creating…" : "Start a new queue"}
          </button>
        </div>
        {startNew.error && (
          <p className="text-sm text-red-600">
            {(startNew.error as Error).message}
          </p>
        )}
      </div>
    );
  }

  const items = itemsInOrder;
  const full = items.length >= QUEUE_CAP;
  const empty = items.length === 0;
  // Ordering needs each song's BPM/key/energy, which exist once the
  // analyze stage has finished (status past "analyzing").
  const ANALYZED_STATUSES = ["analyzed", "separating", "transcribing", "ready"];
  const allAnalyzed =
    items.length >= 3 &&
    items.every((i) => ANALYZED_STATUSES.includes(i.song.status));

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
      <div>
        <h2 className="font-semibold mb-3">Search</h2>
        <QueueSearch
          onAdd={async (r) => {
            await addToQueue.mutateAsync(r);
          }}
          disabled={full || addToQueue.isPending}
        />
      </div>

      <div>
        <GigContext queue={queue.data} />
        <div className="flex items-baseline justify-between mb-3">
          <h2 className="font-semibold">
            Queue
            <span className="ml-2 text-xs opacity-60 tabular-nums">
              {items.length}/{QUEUE_CAP}
            </span>
          </h2>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={() => suggest.mutate()}
              disabled={!allAnalyzed || suggest.isPending}
              title={
                allAnalyzed
                  ? "Reorder the crate to minimize key/tempo/energy friction (keeps your opener)"
                  : items.length < 3
                    ? "Needs at least 3 songs"
                    : "Waiting for song analysis to finish (BPM/key/energy)…"
              }
              className="border rounded px-4 py-2 text-sm hover:bg-black/5 dark:hover:bg-white/10 disabled:opacity-40"
            >
              {suggest.isPending
                ? "Thinking…"
                : !allAnalyzed && items.length >= 3
                  ? "Suggest order (analyzing…)"
                  : "Suggest order"}
            </button>
            <button
              type="button"
              onClick={() => lock.mutate()}
              disabled={empty || lock.isPending}
              className="border rounded px-4 py-2 text-sm hover:bg-black/5 dark:hover:bg-white/10 disabled:opacity-40"
            >
              {lock.isPending ? "Locking…" : "Done"}
            </button>
          </div>
        </div>

        {suggest.error && (
          <p className="text-sm text-amber-600 mb-2">
            {(suggest.error as Error).message}
          </p>
        )}
        {suggestion && (
          <div className="border rounded p-3 mb-3 flex flex-col gap-2 bg-black/5 dark:bg-white/5">
            {suggestion.improved ? (
              <>
                <p className="text-sm">
                  Smoother order found — mix friction{" "}
                  <span className="tabular-nums">
                    {suggestion.current_cost} → {suggestion.suggested_cost}
                  </span>
                </p>
                <ol className="text-xs opacity-80 flex flex-col gap-0.5">
                  {suggestion.order.map((songId, i) => {
                    const item = items.find(
                      (it) => it.song.id === songId
                    );
                    const edge = i > 0 ? suggestion.edges[i - 1] : null;
                    return (
                      <li key={songId} className="truncate">
                        {edge && (
                          <span
                            className={
                              "mr-1 px-1 rounded text-[10px] " +
                              (edge.grade === "A"
                                ? "bg-emerald-500/30"
                                : edge.grade === "B"
                                  ? "bg-amber-500/30"
                                  : "bg-red-500/30")
                            }
                            title={edge.reason}
                          >
                            {edge.grade}
                          </span>
                        )}
                        {i + 1}. {item?.song.title ?? songId}
                      </li>
                    );
                  })}
                </ol>
                <div className="flex gap-2">
                  <button
                    type="button"
                    onClick={applySuggestion}
                    className="border rounded px-3 py-1 text-sm hover:bg-black/5 dark:hover:bg-white/10"
                  >
                    Apply
                  </button>
                  <button
                    type="button"
                    onClick={() => setSuggestion(null)}
                    className="border rounded px-3 py-1 text-sm opacity-70 hover:opacity-100"
                  >
                    Dismiss
                  </button>
                </div>
              </>
            ) : (
              <div className="flex items-center justify-between gap-2">
                <p className="text-sm">
                  Your order is already the smoothest path — nice crate.
                </p>
                <button
                  type="button"
                  onClick={() => setSuggestion(null)}
                  className="border rounded px-3 py-1 text-sm opacity-70 hover:opacity-100"
                >
                  OK
                </button>
              </div>
            )}
          </div>
        )}

        {empty && (
          <p className="text-sm opacity-70">
            Search and add a song to get started.
          </p>
        )}

        <DndContext
          sensors={sensors}
          collisionDetection={closestCenter}
          onDragEnd={onDragEnd}
        >
          <SortableContext
            items={items.map((i) => i.id)}
            strategy={verticalListSortingStrategy}
          >
            <ul className="flex flex-col gap-2">
              {items.map((item) => (
                <QueueItemRow
                  key={item.id}
                  item={item}
                  onRemove={(id) => remove.mutate(id)}
                />
              ))}
            </ul>
          </SortableContext>
        </DndContext>

        {addToQueue.error && (
          <p className="text-sm text-red-600 mt-2">
            {(addToQueue.error as Error).message}
          </p>
        )}
        {lock.error && (
          <p className="text-sm text-red-600 mt-2">
            {(lock.error as Error).message}
          </p>
        )}
      </div>
    </div>
  );
}
