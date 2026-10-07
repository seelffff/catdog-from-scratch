/** Камера через getUserMedia. Работает только на https или localhost. */
export class Camera {
  private stream: MediaStream | null = null;

  constructor(private readonly video: HTMLVideoElement) {}

  get active(): boolean {
    return this.stream !== null;
  }

  async start(): Promise<void> {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('Камера недоступна: нужен https или localhost');
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false });
    } catch {
      throw new Error('Нет доступа к камере');
    }
    this.video.srcObject = this.stream;
    await this.video.play();
  }

  snap(): Promise<Blob> {
    const v = this.video;
    const c = document.createElement('canvas');
    c.width = v.videoWidth;
    c.height = v.videoHeight;
    c.getContext('2d')!.drawImage(v, 0, 0);
    return new Promise((resolve, reject) =>
      c.toBlob(b => (b ? resolve(b) : reject(new Error('Не удалось снять кадр'))), 'image/jpeg', 0.92),
    );
  }

  stop(): void {
    this.stream?.getTracks().forEach(t => t.stop());
    this.stream = null;
    this.video.srcObject = null;
  }
}
