/** Ask the server for the audio of a YouTube link (POST /youtube) and wrap it
 *  as a File, so the rest of the app treats it like any uploaded track. */
export async function fetchYoutubeAudio(url: string): Promise<File> {
  const form = new FormData();
  form.append("url", url);
  const resp = await fetch("/youtube", { method: "POST", body: form });
  if (!resp.ok) {
    let detail: string | undefined;
    try {
      const body = await resp.json();
      if (body && typeof body.detail === "string") detail = body.detail;
    } catch {
      // Non-JSON body: fall through to the generic message.
    }
    throw new Error(detail ?? `Could not fetch the YouTube audio (HTTP ${resp.status}).`);
  }
  const title = decodeURIComponent(resp.headers.get("X-Audio-Title") ?? "") || "youtube";
  const blob = await resp.blob();
  // Strip characters filesystems dislike; the name also becomes the .mid name.
  const safe = title.replace(/[\\/:*?"<>|]+/g, " ").trim() || "youtube";
  return new File([blob], `${safe}.mp3`, { type: "audio/mpeg" });
}
