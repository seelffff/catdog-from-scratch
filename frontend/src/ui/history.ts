import type { Label } from '../api/types';

export interface HistoryEntry {
  id: number;
  blob: Blob;
  /** object URL; им владеет History и отзывает его при вытеснении. */
  url: string;
  source: string;
  result?: { label: Label; confidence: number };
}

const LABEL_TEXT: Record<Label, string> = { cat: 'кошка', dog: 'собака' };

/** Лента последних фото; клик по миниатюре открывает фото снова. */
export class History {
  private entries: HistoryEntry[] = [];
  private seq = 0;
  private currentId: number | null = null;

  constructor(
    private readonly section: HTMLElement,
    private readonly list: HTMLElement,
    private readonly limit: number,
    private readonly onSelect: (entry: HistoryEntry) => void,
  ) {}

  add(blob: Blob, url: string, source: string): HistoryEntry {
    const entry: HistoryEntry = { id: ++this.seq, blob, url, source };
    this.entries.unshift(entry);
    for (const old of this.entries.splice(this.limit)) URL.revokeObjectURL(old.url);
    this.currentId = entry.id;
    this.render();
    return entry;
  }

  setCurrent(id: number): void {
    this.currentId = id;
    this.render();
  }

  setResult(id: number, label: Label, confidence: number): void {
    const e = this.entries.find(x => x.id === id);
    if (!e) return;
    e.result = { label, confidence };
    this.render();
  }

  private render(): void {
    this.section.hidden = this.entries.length < 2;
    this.list.textContent = '';
    for (const e of this.entries) {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'thumb';
      b.classList.toggle('current', e.id === this.currentId);
      b.title = e.source;
      const img = document.createElement('img');
      img.src = e.url;
      img.alt = '';
      const cap = document.createElement('span');
      cap.textContent = e.result ? `${LABEL_TEXT[e.result.label]} ${Math.round(e.result.confidence * 100)}%` : '…';
      b.append(img, cap);
      b.addEventListener('click', () => this.onSelect(e));
      this.list.append(b);
    }
  }
}
