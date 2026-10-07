/** Все способы получить картинку, кроме камеры: файл, drag&drop, вставка, кнопка буфера, ссылка. */

export type MessageKind = 'info' | 'error';

export interface SourceCallbacks {
  onImage(blob: Blob, source: string): void;
  say(text: string, kind?: MessageKind): void;
  setDragging(on: boolean): void;
}

export interface SourceElements {
  fileInput: HTMLInputElement;
  fileButton: HTMLButtonElement;
  pasteButton: HTMLButtonElement;
  urlForm: HTMLFormElement;
  urlInput: HTMLInputElement;
}

const isMac = /Mac|iPhone|iPad/.test(navigator.userAgent);
export const PASTE_SHORTCUT = isMac ? '⌘V' : 'Ctrl V';

const firstUrl = (text: string | undefined) =>
  (text ?? '').split(/\s+/).find(s => /^(https?:|data:image\/)/i.test(s)) ?? null;

const imgSrcFromHtml = (html: string) => /<img[^>]+src=["']([^"']+)["']/i.exec(html)?.[1] ?? null;

export class Sources {
  private dragDepth = 0;

  constructor(
    private readonly el: SourceElements,
    private readonly cb: SourceCallbacks,
  ) {}

  bind(): void {
    const { el, cb } = this;

    el.fileButton.addEventListener('click', () => this.openFilePicker());
    el.fileInput.addEventListener('change', () => {
      const f = el.fileInput.files?.[0];
      if (f) cb.onImage(f, f.name);
      el.fileInput.value = '';
    });

    document.addEventListener('dragenter', e => {
      e.preventDefault();
      this.dragDepth++;
      cb.setDragging(true);
    });
    document.addEventListener('dragleave', () => {
      if (--this.dragDepth <= 0) {
        this.dragDepth = 0;
        cb.setDragging(false);
      }
    });
    document.addEventListener('dragover', e => e.preventDefault());
    document.addEventListener('drop', e => {
      e.preventDefault();
      this.dragDepth = 0;
      cb.setDragging(false);
      if (e.dataTransfer) this.fromTransfer(e.dataTransfer, 'перетащено');
    });

    document.addEventListener('paste', e => {
      const dt = e.clipboardData;
      if (!dt) return;
      const hasImage = [...dt.items].some(i => i.kind === 'file' && i.type.startsWith('image/'));
      if (e.target === el.urlInput && !hasImage) return; // обычная вставка текста в поле ссылки
      e.preventDefault();
      this.fromTransfer(dt, 'из буфера');
    });

    el.pasteButton.addEventListener('click', () => void this.readClipboard());

    el.urlForm.addEventListener('submit', e => {
      e.preventDefault();
      const v = el.urlInput.value.trim();
      if (v) void this.loadUrl(v);
    });
  }

  openFilePicker(): void {
    this.el.fileInput.click();
  }

  // Всё читаем из DataTransfer синхронно: после await браузер его очищает.
  private fromTransfer(dt: DataTransfer, source: string): void {
    const file = [...dt.files].find(f => f.type.startsWith('image/'));
    if (file) return this.cb.onImage(file, source);
    const item = [...dt.items].find(i => i.kind === 'file' && i.type.startsWith('image/'));
    const asFile = item?.getAsFile();
    if (asFile) return this.cb.onImage(asFile, source);
    const url =
      imgSrcFromHtml(dt.getData('text/html')) ?? firstUrl(dt.getData('text/uri-list')) ?? firstUrl(dt.getData('text/plain'));
    if (url) return void this.loadUrl(url);
    this.cb.say('Здесь нет изображения', 'error');
  }

  private async readClipboard(): Promise<void> {
    if (!navigator.clipboard?.read) {
      this.cb.say(`Нажмите ${PASTE_SHORTCUT}: браузер не даёт читать буфер по кнопке`);
      return;
    }
    try {
      for (const item of await navigator.clipboard.read()) {
        const type = item.types.find(t => t.startsWith('image/'));
        if (type) return this.cb.onImage(await item.getType(type), 'из буфера');
        if (item.types.includes('text/plain')) {
          const url = firstUrl(await (await item.getType('text/plain')).text());
          if (url) return void this.loadUrl(url);
        }
      }
      this.cb.say('В буфере нет картинки', 'error');
    } catch {
      this.cb.say(`Нет доступа к буферу. Нажмите ${PASTE_SHORTCUT}.`, 'error');
    }
  }

  async loadUrl(raw: string): Promise<void> {
    let url: URL;
    try {
      url = new URL(raw.trim(), location.href);
    } catch {
      return this.cb.say('Неверная ссылка', 'error');
    }
    this.cb.say('загружаю…');
    try {
      const res = await fetch(url.href);
      if (!res.ok) throw new Error(String(res.status));
      this.cb.onImage(await res.blob(), url.protocol === 'data:' ? 'по ссылке' : url.hostname);
    } catch {
      this.cb.say(
        'Сайт не даёт скачать картинку напрямую. Скопируйте само изображение (правый клик → «Копировать картинку») и вставьте.',
        'error',
      );
    }
  }
}
