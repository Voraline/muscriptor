export async function FetchYoutubeAudio(Url: string): Promise<File> {
  const Payload = new FormData();
  Payload.append("url", Url);
  const Resp = await fetch("/youtube", { method: "POST", body: Payload });
  if (!Resp.ok) {
    let Detail: string | undefined;
    try {
      const Body = await Resp.json();
      if (Body && typeof Body.detail === "string") Detail = Body.detail;
    } catch {
      Detail = undefined;
    }
    throw new Error(Detail ?? `Could not fetch the YouTube audio (HTTP ${Resp.status}).`);
  }
  const Title = decodeURIComponent(Resp.headers.get("X-Audio-Title") ?? "") || "youtube";
  const AudioBlob = await Resp.blob();
  const Safe = Title.replace(/[\\/:*?"<>|]+/g, " ").trim() || "youtube";
  return new File([AudioBlob], `${Safe}.mp3`, { type: "audio/mpeg" });
}
