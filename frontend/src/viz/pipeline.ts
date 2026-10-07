import type { Activations } from '../net/cnn';
import type { FeatureMap } from '../net/tensor';
import { clearCanvas, paintMap, type PaintOptions } from './paint';

export const STAGES = [
  { id: 'input', n: '01', title: 'Вход', desc: 'Фото сжато до 64×64 и переведено в яркость.' },
  { id: 'conv1', n: '02', title: 'Свёртка', desc: 'Фильтры 3×3 ищут края и пятна.' },
  { id: 'relu1', n: '03', title: 'ReLU', desc: 'Отрицательные отклики обнуляются.' },
  { id: 'pool1', n: '04', title: 'Пулинг', desc: 'Из каждого квадрата 2×2 остаётся максимум.' },
  { id: 'conv2', n: '05', title: 'Свёртка 2', desc: 'Простые признаки складываются в формы.' },
  { id: 'pool2', n: '06', title: 'Пулинг', desc: 'Меньше деталей, больше смысла.' },
] as const;

const ORDER: readonly Exclude<ActiveId, null>[] = [...STAGES.map(s => s.id), 'dense'];

export type StageId = (typeof STAGES)[number]['id'];
export type ActiveId = StageId | 'dense' | null;

const BLANK: Record<StageId, readonly [size: number, count: number]> = {
  input: [64, 1], conv1: [64, 4], relu1: [64, 4], pool1: [32, 4], conv2: [32, 4], pool2: [16, 4],
};
const MAX_MAPS = 9;
const STAGE_PAINT: Partial<Record<StageId, PaintOptions>> = { input: { photo: true }, conv1: { signed: true } };

export const mapsOf = (acts: Activations, id: StageId): FeatureMap[] => (id === 'input' ? [acts.input] : acts[id]);

export interface MapCell {
  wrap: HTMLDivElement;
  canvas: HTMLCanvasElement;
  scan: HTMLSpanElement;
  index: number;
}

export interface PipelineHandlers {
  /** Курсор над входом или картами первого слоя; координаты 0..1. */
  onHover(fx: number, fy: number): void;
  onLeave(): void;
  onPickFilter(index: number): void;
  filterCount: number;
}

/** Ряд карт признаков от входа до последнего pooling. */
export class Pipeline {
  cells = {} as Record<StageId, MapCell[]>;

  constructor(
    private readonly root: HTMLElement,
    private readonly handlers: PipelineHandlers,
  ) {}

  build(acts: Activations | null): void {
    this.root.textContent = '';
    for (const stage of STAGES) {
      const shapes = acts
        ? mapsOf(acts, stage.id).map(m => [m.w, m.h] as const)
        : Array.from({ length: BLANK[stage.id][1] }, () => [BLANK[stage.id][0], BLANK[stage.id][0]] as const);
      const shown = shapes.slice(0, MAX_MAPS);

      const li = document.createElement('li');
      li.className = 'stage';
      li.dataset.stage = stage.id;
      const more = shapes.length > MAX_MAPS ? ` · показаны ${MAX_MAPS}` : '';
      const isReal = stage.id !== 'input' && (acts?.fromModel.has(stage.id) ?? false);
      const actualDescriptions: Partial<Record<StageId, string>> = {
        conv1: 'Настоящая свёртка 3×3: показаны 4 из 32 каналов, уменьшенные для просмотра.',
        relu1: 'Настоящий ReLU после BatchNorm; показаны первые четыре канала.',
        pool1: 'Настоящий max pooling 2×2 после третьей свёртки; показаны четыре уменьшенные карты.',
      };
      const description = isReal ? actualDescriptions[stage.id] ?? stage.desc
        : stage.id === 'input' ? 'Учебный вход 64×64 в оттенках серого. Модель получает RGB 256×256.'
        : `Учебная иллюстрация. ${stage.desc}`;
      li.innerHTML = `
        <div class="stage-rail"></div>
        <div class="stage-head"><span class="stage-n">${stage.n}</span><span class="stage-title">${stage.title}</span></div>
        <div class="maps" style="--cols:${Math.ceil(Math.sqrt(shown.length))}"></div>
        <span class="stage-dims">${shapes[0][0]}×${shapes[0][1]}×${shapes.length}${more}</span>
        <p class="stage-desc">${description}</p>`;
      const grid = li.querySelector<HTMLDivElement>('.maps')!;

      this.cells[stage.id] = shown.map(([w, h], index) => {
        const wrap = document.createElement('div');
        wrap.className = 'map';
        const canvas = document.createElement('canvas');
        canvas.width = w;
        canvas.height = h;
        const scan = document.createElement('span');
        scan.className = 'scan';
        scan.hidden = true;
        wrap.append(canvas, scan);
        grid.append(wrap);
        clearCanvas(canvas);
        return { wrap, canvas, scan, index };
      });
      this.root.append(li);
    }
    this.wire();
  }

  private wire(): void {
    const { onHover, onLeave, onPickFilter, filterCount } = this.handlers;
    for (const cell of [...this.cells.input, ...this.cells.conv1, ...this.cells.relu1]) {
      cell.wrap.addEventListener('pointermove', e => {
        const r = cell.wrap.getBoundingClientRect();
        onHover((e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height);
      });
      cell.wrap.addEventListener('pointerleave', onLeave);
    }
    for (const cell of this.cells.conv1) {
      if (cell.index >= filterCount) continue;
      cell.wrap.classList.add('pick');
      cell.wrap.addEventListener('click', () => onPickFilter(cell.index));
    }
  }

  paint(acts: Activations, id: StageId, extra: PaintOptions = {}): void {
    const maps = mapsOf(acts, id);
    for (const c of this.cells[id]) paintMap(c.canvas, maps[c.index], { ...STAGE_PAINT[id], ...extra });
  }

  reveal(acts: Activations, id: StageId, t: number): void {
    const maps = mapsOf(acts, id);
    for (const c of this.cells[id]) {
      const m = maps[c.index];
      paintMap(c.canvas, m, { ...STAGE_PAINT[id], reveal: Math.floor(t * m.w * m.h) });
    }
  }

  /** ReLU показывается как исчезновение красного (минуса) с карт первой свёртки. */
  paintRelu(acts: Activations, t: number): void {
    for (const c of this.cells.relu1) {
      const before = acts.conv1[c.index];
      const after = acts.relu1[c.index];
      const sameShape = before && before.w === after.w && before.h === after.h;
      if (sameShape && !acts.fromModel.has('relu1')) paintMap(c.canvas, before, { signed: true, reluFade: t });
      else paintMap(c.canvas, after, { reveal: Math.floor(t * after.w * after.h) });
    }
  }

  paintStageFinal(acts: Activations, id: StageId): void {
    if (id === 'relu1') this.paintRelu(acts, 1);
    else this.paint(acts, id);
  }

  clear(): void {
    for (const cell of Object.values(this.cells).flat()) clearCanvas(cell.canvas);
  }

  showScan(cell: MapCell, x: number, y: number, m: { w: number; h: number }): void {
    const s = cell.scan;
    s.hidden = false;
    s.style.left = `${((x - 1) / m.w) * 100}%`;
    s.style.top = `${((y - 1) / m.h) * 100}%`;
    s.style.width = `${(3 / m.w) * 100}%`;
    s.style.height = `${(3 / m.h) * 100}%`;
  }

  hideScans(id?: StageId): void {
    const cells = id ? this.cells[id] : Object.values(this.cells).flat();
    for (const c of cells) c.scan.hidden = true;
  }

  markFilter(index: number): void {
    for (const c of this.cells.conv1 ?? []) c.wrap.classList.toggle('selected', c.index === index);
  }
}

/** Подсвечивает текущий шаг; все шаги до него отмечаются пройденными. null — всё пройдено. */
export function setActiveStage(id: ActiveId): void {
  const current = id === null ? ORDER.length : ORDER.indexOf(id);
  document.querySelectorAll<HTMLElement>('[data-stage]').forEach(el => {
    const i = ORDER.indexOf(el.dataset.stage as Exclude<ActiveId, null>);
    el.classList.toggle('active', i === current);
    el.classList.toggle('done', i < current);
  });
}

export function resetStages(): void {
  document.querySelectorAll<HTMLElement>('[data-stage]').forEach(el => el.classList.remove('active', 'done'));
}
