/** Квадратная рамка: пусто, фото или видео с камеры. */
export class Frame {
  constructor(
    readonly root: HTMLElement,
    private readonly empty: HTMLElement,
    private readonly preview: HTMLImageElement,
    private readonly video: HTMLVideoElement,
  ) {}

  showImage(url: string): void {
    this.preview.src = url;
    this.set('image');
  }

  showCamera(): void {
    this.set('camera');
  }

  /** После закрытия камеры: вернуть последнее фото или пустое состояние. */
  restore(): void {
    this.set(this.preview.getAttribute('src') ? 'image' : 'empty');
  }

  setDragging(on: boolean): void {
    this.root.classList.toggle('dragging', on);
  }

  private set(view: 'empty' | 'image' | 'camera'): void {
    this.empty.hidden = view !== 'empty';
    this.preview.hidden = view !== 'image';
    this.video.hidden = view !== 'camera';
  }
}
