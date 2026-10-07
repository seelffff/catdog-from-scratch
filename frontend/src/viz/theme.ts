/** Цвета для canvas берутся из CSS-переменных (src/styles/tokens.css), так тема остаётся в одном месте. */

export type RGB = readonly [number, number, number];

export interface Palette {
  mapBg: RGB;
  surface: RGB;
  ink: RGB;
  accent: RGB;
  muted: RGB;
  sans: string;
  mono: string;
}

function hexToRgb(hex: string): RGB {
  let h = hex.trim().replace('#', '');
  if (h.length === 3) h = [...h].map(c => c + c).join('');
  const n = parseInt(h, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function read(): Palette {
  const cs = getComputedStyle(document.documentElement);
  const v = (name: string) => cs.getPropertyValue(name).trim();
  return {
    mapBg: hexToRgb(v('--map-bg')),
    surface: hexToRgb(v('--surface')),
    ink: hexToRgb(v('--ink')),
    accent: hexToRgb(v('--accent')),
    muted: hexToRgb(v('--muted')),
    sans: v('--sans'),
    mono: v('--mono'),
  };
}

let current: Palette | null = null;

export const palette = (): Palette => (current ??= read());

const THEME_EVENT = 'themechange';

export function onThemeChange(cb: () => void): void {
  const handler = () => {
    current = read();
    cb();
  };
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', handler);
  document.addEventListener(THEME_EVENT, handler);
}

export type ThemeName = 'light' | 'dark';

export function effectiveTheme(): ThemeName {
  const explicit = document.documentElement.dataset.theme;
  if (explicit === 'light' || explicit === 'dark') return explicit;
  return matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

/** Явный выбор темы: запоминается в localStorage, перерисовывает canvas. */
export function setTheme(theme: ThemeName): void {
  document.documentElement.dataset.theme = theme;
  try {
    localStorage.setItem('theme', theme);
  } catch {
    /* приватный режим — просто не запоминаем */
  }
  document.dispatchEvent(new Event(THEME_EVENT));
}

export const mix = (a: RGB, b: RGB, t: number): RGB => [
  a[0] + (b[0] - a[0]) * t,
  a[1] + (b[1] - a[1]) * t,
  a[2] + (b[2] - a[2]) * t,
];

export const rgba = (c: RGB, a = 1) => `rgba(${c[0] | 0},${c[1] | 0},${c[2] | 0},${a})`;
