import { useState } from "react";
import { Button } from "./Button";

export function YoutubeInput(Props: { OnSubmit: (Url: string) => Promise<void> }) {
  const [Url, SetUrl] = useState("");
  const [Loading, SetLoading] = useState(false);

  async function Go() {
    if (!Url.trim() || Loading) return;
    SetLoading(true);
    try {
      await Props.OnSubmit(Url.trim());
      SetUrl("");
    } finally {
      SetLoading(false);
    }
  }

  return (
    <div className="flex w-full max-w-xl items-center gap-2">
      <input
        type="url"
        value={Url}
        disabled={Loading}
        placeholder="Paste a YouTube link"
        aria-label="YouTube link"
        onChange={(e) => SetUrl(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") Go();
        }}
        className="min-w-0 flex-1 rounded-xl border border-white/15 bg-transparent px-4 py-2.5 text-base text-content placeholder:text-muted"
      />
      <Button onClick={Go} disabled={Loading || !Url.trim()}>
        {Loading ? "Fetching audio…" : "Use link"}
      </Button>
    </div>
  );
}
