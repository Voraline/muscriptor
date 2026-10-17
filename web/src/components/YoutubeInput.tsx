import { useState } from "react";
import { Button } from "./Button";

/** A YouTube link field. `onSubmit` resolves once the audio has been fetched. */
export function YoutubeInput(props: { onSubmit: (url: string) => Promise<void> }) {
  const [url, setUrl] = useState("");
  const [loading, setLoading] = useState(false);

  async function go() {
    if (!url.trim() || loading) return;
    setLoading(true);
    try {
      await props.onSubmit(url.trim());
      setUrl("");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex w-full max-w-xl items-center gap-2">
      <input
        type="url"
        value={url}
        disabled={loading}
        placeholder="Paste a YouTube link"
        aria-label="YouTube link"
        onChange={(e) => setUrl(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") go();
        }}
        className="min-w-0 flex-1 rounded-xl border border-white/15 bg-transparent px-4 py-2.5 text-base text-content placeholder:text-muted"
      />
      <Button onClick={go} disabled={loading || !url.trim()}>
        {loading ? "Fetching audio…" : "Use link"}
      </Button>
    </div>
  );
}
