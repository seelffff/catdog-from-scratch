import { FILTERS, convStep } from '../net/cnn';
import type { FeatureMap } from '../net/tensor';
import { mix, palette, rgba } from './theme';

const fmt = (v: number) => (v < 0 ? `−${-v}` : `${v}`);

/** Панель «Свёртка крупным планом»: один шаг свёртки с числами. */
export class Closeup {
  private filter = 0;
  private x = 32;
  private y = 32;
  private input: FeatureMap | null = null;
  private readonly tabs: HTMLButtonElement[];
  private readonly patchCells: HTMLSpanElement[];
  private readonly kernelCells: HTMLSpanElement[];
  private readonly sumEl = byId('gSum');
  private readonly reluEl = byId('gRelu');
  private readonly formulaEl = byId('formula');
  private readonly posEl = byId('closeupPos');

  constructor(private readonly onFilterChange: (index: number) => void) {
    const tabsRoot = byId('kernelTabs');
    this.tabs = FILTERS.map((f, i) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.textContent = f.name;
      b.addEventListener('click', () => this.setFilter(i));
      tabsRoot.append(b);
      return b;
    });
    this.patchCells = cells('gPatch');
    this.kernelCells = cells('gKernel');
    this.markTabs();
    this.render();
  }

  get filterIndex(): number {
    return this.filter;
  }

  setInput(m: FeatureMap | null): void {
    this.input = m;
    this.render();
  }

  setFilter(i: number): void {
    this.filter = i;
    this.markTabs();
    this.onFilterChange(i);
    this.render();
  }

  moveTo(x: number, y: number): void {
    if (x === this.x && y === this.y) return;
    this.x = x;
    this.y = y;
    this.render();
  }

  private markTabs(): void {
    this.tabs.forEach((b, i) => b.classList.toggle('on', i === this.filter));
  }

  render(): void {
    const { surface, ink, accent } = palette();
    const kernel = FILTERS[this.filter].kernel;
    const kMax = Math.max(...kernel.map(Math.abs));
    this.kernelCells.forEach((c, i) => {
      const k = kernel[i];
      const a = (Math.abs(k) / kMax) * 0.85;
      c.textContent = fmt(k);
      c.style.background = rgba(mix(surface, k >= 0 ? ink : accent, a));
      c.style.color = a > 0.45 ? rgba(surface) : rgba(ink);
    });

    if (!this.input) {
      for (const c of this.patchCells) {
        c.textContent = '';
        c.style.background = '';
      }
      this.sumEl.textContent = this.reluEl.textContent = '—';
      this.sumEl.style.color = '';
      this.formulaEl.textContent = 'Загрузите фото: здесь появится расчёт одного шага свёртки.';
      this.posEl.textContent = '';
      return;
    }

    const step = convStep(this.input, this.x, this.y, this.filter);
    this.patchCells.forEach((c, i) => {
      const v = step.pixels[i];
      c.textContent = String(v);
      c.style.background = `rgb(${v},${v},${v})`;
      c.style.color = v > 140 ? '#151515' : '#f2f0eb';
    });
    this.sumEl.textContent = fmt(step.sum);
    this.sumEl.style.color = step.sum < 0 ? rgba(accent) : '';
    this.reluEl.textContent = String(step.relu);
    const terms = step.pixels
      .map((v, i) => [v, step.kernel[i]] as const)
      .filter(([, k]) => k !== 0)
      .map(([v, k]) => `${v}·${k < 0 ? `(${fmt(k)})` : k}`);
    this.formulaEl.textContent = `${terms.join(' + ')} = ${fmt(step.sum)}`;
    this.posEl.textContent = `точка x=${step.x}, y=${step.y} · фильтр «${step.filter.name}» · нулевые веса не показаны`;
  }
}

function byId<T extends HTMLElement = HTMLElement>(id: string): T {
  const el = document.getElementById(id);
  if (!el) throw new Error(`#${id} не найден в разметке`);
  return el as T;
}

function cells(id: string): HTMLSpanElement[] {
  const grid = byId(id);
  return Array.from({ length: 9 }, () => grid.appendChild(document.createElement('span')));
}
