export interface Segment {
  text: string;
  match: boolean;
}

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** Splits `text` into plain and matched runs for every whitespace-separated query term, case-insensitively. */
export function highlightSegments(text: string, query: string): Segment[] {
  const terms = query.split(/\s+/).filter(Boolean).map(escapeRegExp);
  if (!terms.length) return [{ text, match: false }];
  // A capturing split alternates plain, match, plain, ...; tag before dropping empty runs.
  return text
    .split(new RegExp(`(${terms.join("|")})`, "iu"))
    .map((part, i) => ({ text: part, match: i % 2 === 1 }))
    .filter((s) => s.text !== "");
}
