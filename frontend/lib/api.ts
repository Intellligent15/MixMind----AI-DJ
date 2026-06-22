// Browser-side API base. Server Components use BACKEND_URL instead — that
// resolves inside the docker network, this one resolves from the user's
// browser.
const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export type SongStatus =
  | "pending"
  | "downloading"
  | "downloaded"
  | "analyzing"
  | "analyzed"
  | "separating"
  | "transcribing"
  | "ready"
  | "failed";

export type SearchResult = {
  youtube_video_id: string;
  title: string;
  artist: string | null;
  duration_seconds: number;
  thumbnail_url: string | null;
};

export type Song = {
  id: string;
  youtube_video_id: string;
  title: string;
  artist: string | null;
  duration_seconds: number;
  thumbnail_url: string | null;
  audio_path: string | null;
  status: SongStatus;
  // Reason the song last failed (null otherwise). Surfaced + retryable in
  // the Processing view.
  error_text: string | null;
  created_at: string;
  updated_at: string;
  has_stems: boolean;
  has_transcription: boolean;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${body}`);
  }
  // 204 No Content (and any empty body) — don't try to parse JSON.
  if (res.status === 204) {
    return undefined as T;
  }
  return res.json() as Promise<T>;
}

export type Section = {
  start: number;
  end: number;
  label: string;
};

export type Analysis = {
  id: string;
  song_id: string;
  bpm: number;
  key: string;
  camelot_key: string;
  time_signature: number;
  beat_grid: number[];
  downbeats: number[];
  sections: Section[];
  energy_curve: number[];
  vocal_segments: number[][];
  created_at: string;
  updated_at: string;
};

export const STEM_NAMES = ["vocals", "drums", "bass", "other"] as const;
export type StemName = (typeof STEM_NAMES)[number];

export type StemsStatus = "pending" | "separating" | "separated" | "failed";

export type Stems = {
  id: string;
  song_id: string;
  model_name: string;
  status: StemsStatus;
  vocals_path: string | null;
  drums_path: string | null;
  bass_path: string | null;
  other_path: string | null;
  vocal_rms: number | null;
  created_at: string;
  updated_at: string;
};

export type TranscriptionStatus =
  | "not_attempted"
  | "success"
  | "skipped_instrumental"
  | "error";

export type TranscriptionWord = {
  start: number;
  end: number;
  word: string;
};

export type TranscriptionSegment = {
  start: number;
  end: number;
  text: string;
  words: TranscriptionWord[];
};

export type Transcription = {
  id: string;
  song_id: string;
  model_name: string;
  status: TranscriptionStatus;
  language: string | null;
  segments: TranscriptionSegment[];
  vocal_rms_threshold: number | null;
  vocal_rms_observed: number | null;
  duration_seconds: number | null;
  created_at: string;
  updated_at: string;
};

export type QueueItem = {
  id: string;
  queue_id: string;
  position: number;
  // Whole-song semitone offset (pitch_mode="whole_song"): this song
  // plays shifted by this many semitones for its entire duration in
  // this queue's mix. 0 = native key.
  pitch_offset_semitones: number;
  song: Song;
};

export type Queue = {
  id: string;
  locked: boolean;
  created_at: string;
  locked_at: string | null;
  items: QueueItem[];
};

export type MixPlanStatus = "pending" | "rendering" | "ready" | "failed";

export type MixPlan = {
  id: string;
  queue_id: string;
  from_song_id: string;
  to_song_id: string;
  plan_json: Record<string, unknown>[] | null;
  rendered_audio_path: string | null;
  status: MixPlanStatus;
  error_text: string | null;
  // Planner v2 telemetry & controls
  plan_source: string | null;
  style: string | null;
  rationale: string | null;
  style_hint: string | null;
  style_override: string | null;
  reroll_nonce: number;
  created_at: string;
  updated_at: string;
};

export type TransitionStyleInfo = {
  id: string;
  description: string;
  duration_choices: number[];
};

export type VocalSafeRegion = {
  start: number;
  end: number;
  safe: boolean;
  reason: string;
};

export type VocalSafeRegionsResponse = {
  regions: VocalSafeRegion[];
};

export type QueueRenderStatus = "pending" | "rendering" | "ready" | "failed";

// Stitched-mix output timeline (Phase 10). Times are seconds in the
// continuous-mix playback timeline, so they map directly onto the player's
// playhead position.
export type MixTimelineSong = {
  index: number;
  song_id: string;
  title: string | null;
  artist: string | null;
  start: number;
  end: number;
};

export type MixTimelineStemRoute = {
  stem: string | null;
  from: string | null; // "A" | "B"
  to: string | null; // "A" | "B"
};

export type MixTimelineTransition = {
  index: number;
  from_song_id: string;
  to_song_id: string;
  start: number;
  end: number;
  label: string; // human-readable technique, e.g. "Filter sweep"
  stems: MixTimelineStemRoute[];
  effects: string[];
  reasoning: string | null;
};

export type MixTimeline = {
  duration: number;
  songs: MixTimelineSong[];
  transitions: MixTimelineTransition[];
};

export type QueueRender = {
  id: string;
  queue_id: string;
  rendered_audio_path: string | null;
  status: QueueRenderStatus;
  error_text: string | null;
  timeline: MixTimeline | null;
  created_at: string;
  updated_at: string;
};

export const api = {
  search: (q: string, limit = 10) =>
    request<SearchResult[]>(
      `/api/search?q=${encodeURIComponent(q)}&limit=${limit}`
    ),
  listSongs: () => request<Song[]>(`/api/songs`),
  createSong: (payload: Omit<SearchResult, never>) =>
    request<Song>(`/api/songs`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  getSong: (id: string) => request<Song>(`/api/songs/${id}`),
  deleteSong: (id: string) =>
    request<void>(`/api/songs/${id}`, { method: "DELETE" }),
  audioUrl: (id: string) => `${API_BASE}/api/songs/${id}/audio`,
  triggerAnalyze: (id: string) =>
    request<Song>(`/api/songs/${id}/analyze`, { method: "POST" }),
  retrySong: (id: string) =>
    request<Song>(`/api/songs/${id}/retry`, { method: "POST" }),
  getAnalysis: (id: string) => request<Analysis>(`/api/songs/${id}/analysis`),

  triggerSeparate: (id: string) =>
    request<Song>(`/api/songs/${id}/separate`, { method: "POST" }),
  getStems: (id: string) => request<Stems>(`/api/songs/${id}/stems`),
  stemAudioUrl: (id: string, name: StemName) =>
    `${API_BASE}/api/songs/${id}/stems/${name}`,

  triggerTranscribe: (id: string) =>
    request<Song>(`/api/songs/${id}/transcribe`, { method: "POST" }),
  getTranscription: (id: string) =>
    request<Transcription>(`/api/songs/${id}/transcription`),
  getLyrics: (id: string) =>
    request<any>(`/api/songs/${id}/lyrics`),
  getVocalSafeRegions: (id: string) =>
    request<VocalSafeRegionsResponse>(`/api/songs/${id}/vocal_safe_regions`),

  getCurrentQueue: () => request<Queue>(`/api/queues/current`),
  createQueue: () => request<Queue>(`/api/queues`, { method: "POST" }),
  addToQueue: (queueId: string, songId: string) =>
    request<Queue>(`/api/queues/${queueId}/items`, {
      method: "POST",
      body: JSON.stringify({ song_id: songId }),
    }),
  removeFromQueue: (queueId: string, itemId: string) =>
    request<Queue>(`/api/queues/${queueId}/items/${itemId}`, {
      method: "DELETE",
    }),
  reorderQueue: (queueId: string, orderedItemIds: string[]) =>
    request<Queue>(`/api/queues/${queueId}/items`, {
      method: "PATCH",
      body: JSON.stringify({ ordered_item_ids: orderedItemIds }),
    }),
  lockQueue: (queueId: string) =>
    request<Queue>(`/api/queues/${queueId}/lock`, { method: "POST" }),

  getMixPlan: async (id: string): Promise<MixPlan | null> => {
    try {
      return await request<MixPlan>(`/api/mix_plans/${id}`);
    } catch (err) {
      if (isStatusError(err, 404)) return null;
      throw err;
    }
  },
  listMixPlansForQueue: (queueId: string) =>
    request<MixPlan[]>(`/api/queues/${queueId}/mix_plans`),
  triggerRenderMixPlan: (id: string) =>
    request<void>(`/api/mix_plans/${id}/render`, { method: "POST" }),
  // Throw away a pair's plan and generate a fresh one. Pass a style id to
  // pin the transition archetype, or "auto" / undefined to let the LLM
  // choose. Re-renders the pair and re-stitches the mix.
  rerollMixPlan: (id: string, style?: string) =>
    request<{ status: string; reroll_nonce: number }>(
      `/api/mix_plans/${id}/reroll`,
      { method: "POST", body: JSON.stringify({ style: style ?? null }) }
    ),
  listTransitionStyles: () =>
    request<TransitionStyleInfo[]>(`/api/transition_styles`),
  mixPlanAudioUrl: (id: string) => `${API_BASE}/api/mix_plans/${id}/audio`,

  stitchQueue: (queueId: string) =>
    request<{message: string}>(`/api/queues/${queueId}/stitch`, { method: "POST" }),
  getQueueMix: async (queueId: string): Promise<QueueRender | null> => {
    try {
      return await request<QueueRender>(`/api/queues/${queueId}/mix`);
    } catch (err) {
      if (isStatusError(err, 404)) return null;
      throw err;
    }
  },
  queueMixAudioUrl: (queueId: string) => `${API_BASE}/api/queues/${queueId}/mix/audio`,
};

// Distinguishes a `request()` 4xx/5xx from a network error so callers can
// branch on status (e.g. /current returning 404 means "no queue yet").
export function isStatusError(err: unknown, code: number): boolean {
  return err instanceof Error && err.message.startsWith(`${code} `);
}
