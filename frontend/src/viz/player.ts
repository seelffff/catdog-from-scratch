import type { Prediction } from '../api/types';
import type { Activations } from '../net/cnn';
import type { Closeup } from './closeup';
import type { DenseView } from './dense';
import { mapsOf, setActiveStage, type Pipeline, type StageId } from './pipeline';

const CANCEL = Symbol('cancel');
const nextFrame = () => new Promise<number>(r => requestAnimationFrame(r));
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');

/** Сценарий анимации: слой за слоем, со сканированием свёрток и импульсами в dense. */
export class Player {
  /** Идёт сканирование свёртки: наведение курсора в это время не перехватывает рамку. */
  scanning = false;
  acts: Activations | null = null;
  private runId = 0;
  private skipping = false;
  private painted = new Set<StageId>();

  constructor(
    private readonly pipeline: Pipeline,
    private readonly closeup: Closeup,
    private readonly dense: DenseView,
  ) {}

  /** Сразу показать вход, пока модель думает. */
  prepare(local: Activations): void {
    this.cancel();
    this.acts = local;
    this.pipeline.build(null);
    this.pipeline.markFilter(this.closeup.filterIndex);
    this.pipeline.paint(local, 'input');
    this.painted = new Set(['input']);
    setActiveStage('input');
    this.dense.clear();
  }

  cancel(): void {
    this.runId++;
    this.scanning = false;
  }

  skip(): void {
    this.skipping = true;
  }

  /** Перерисовать готовые слои (например, после смены темы). */
  repaint(): void {
    if (this.acts) for (const id of this.painted) this.pipeline.paintStageFinal(this.acts, id);
    this.dense.draw();
    this.closeup.render();
  }

  /** true — анимация дошла до конца, false — её прервали новым запуском. */
  async run(acts: Activations, pred: Prediction): Promise<boolean> {
    const id = ++this.runId;
    this.skipping = reducedMotion.matches;
    this.acts = acts;
    this.painted = new Set();
    const { pipeline: pl, dense } = this;

    const alive = () => {
      if (id !== this.runId) throw CANCEL;
    };
    const tween = async (ms: number, fn: (t: number) => void) => {
      const t0 = performance.now();
      for (;;) {
        alive();
        const t = this.skipping ? 1 : Math.min(1, (performance.now() - t0) / ms);
        fn(t);
        if (t >= 1) return;
        await nextFrame();
      }
    };
    const stage = async (sid: StageId, body: () => Promise<void>) => {
      setActiveStage(sid);
      await body();
      this.painted.add(sid);
    };

    const scan = async (srcId: StageId, dstId: StageId, ms: number, withCloseup: boolean) => {
      const src = mapsOf(acts, srcId);
      const dst = mapsOf(acts, dstId);
      const W = dst[0].w;
      const H = dst[0].h;
      const total = W * H;
      this.scanning = true;
      await tween(ms, t => {
        const idx = Math.min(total, Math.floor(t * total));
        pl.paint(acts, dstId, { reveal: idx });
        if (idx >= total) return;
        const x = idx % W;
        const y = Math.floor(idx / W);
        for (const cell of pl.cells[srcId]) {
          const m = src[cell.index];
          pl.showScan(cell, Math.floor((x * m.w) / W), Math.floor((y * m.h) / H), m);
        }
        if (withCloseup) {
          this.closeup.moveTo(Math.floor((x * acts.input.w) / W), Math.floor((y * acts.input.h) / H));
        }
      });
      pl.hideScans(srcId);
      this.scanning = false;
    };

    pl.build(acts);
    pl.markFilter(this.closeup.filterIndex);
    dense.setData(acts, pred);
    dense.progress = 0;
    dense.draw();

    try {
      await stage('input', async () => {
        pl.paint(acts, 'input');
        await tween(300, () => {});
      });
      await stage('conv1', () => scan('input', 'conv1', 2400, true));
      await stage('relu1', () => tween(700, t => pl.paintRelu(acts, t)));
      await stage('pool1', () => tween(500, t => pl.reveal(acts, 'pool1', t)));
      await stage('conv2', () => scan('pool1', 'conv2', 1500, false));
      await stage('pool2', () => tween(400, t => pl.reveal(acts, 'pool2', t)));
      setActiveStage('dense');
      await tween(1900, t => {
        dense.progress = t * 2;
        dense.draw();
      });
      setActiveStage(null);
      return true;
    } catch (e) {
      if (e === CANCEL) return false;
      throw e;
    } finally {
      if (id === this.runId) {
        this.scanning = false;
        pl.hideScans();
      }
    }
  }
}
