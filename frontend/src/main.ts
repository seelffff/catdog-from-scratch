import '@fontsource/ibm-plex-sans/400.css';
import '@fontsource/ibm-plex-sans/500.css';
import '@fontsource/ibm-plex-mono/400.css';
import '@fontsource/ibm-plex-mono/500.css';
import '@fontsource/cormorant-garamond/500.css';
import '@fontsource/cormorant-garamond/500-italic.css';
import './styles/tokens.css';
import './styles/app.css';

import { ModelClient } from './api/client';
import type { Prediction } from './api/types';
import { config } from './config';
import { FILTERS, computeFromImage, mergeActivations, type Activations } from './net/cnn';
import { Camera } from './input/camera';
import { PASTE_SHORTCUT, Sources, type MessageKind } from './input/sources';
import { Frame } from './ui/frame';
import { History, type HistoryEntry } from './ui/history';
import { renderStatus } from './ui/status';
import { Verdict } from './ui/verdict';
import { Closeup } from './viz/closeup';
import { DenseView } from './viz/dense';
import { Pipeline } from './viz/pipeline';
import { Player } from './viz/player';
import { effectiveTheme, onThemeChange, setTheme } from './viz/theme';

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

/* ---------- компоненты ---------- */

const client = new ModelClient();
const verdict = new Verdict();
const frame = new Frame($('frame'), $('frameEmpty'), $<HTMLImageElement>('preview'), $<HTMLVideoElement>('camera'));
const camera = new Camera($<HTMLVideoElement>('camera'));
const msg = $('msg');
const btnCamera = $<HTMLButtonElement>('btnCamera');
const btnCameraLabel = $('btnCameraLabel');
const btnReplay = $<HTMLButtonElement>('btnReplay');
const btnSkip = $<HTMLButtonElement>('btnSkip');

const say = (text: string, kind: MessageKind = 'info') => {
  msg.textContent = text;
  msg.dataset.kind = kind;
};

const closeup = new Closeup(i => pipeline.markFilter(i));
const pipeline = new Pipeline($('pipeline'), {
  filterCount: FILTERS.length,
  onHover(fx, fy) {
    const input = player.acts?.input;
    if (!input || player.scanning) return;
    const x = Math.min(input.w - 1, Math.max(0, Math.floor(fx * input.w)));
    const y = Math.min(input.h - 1, Math.max(0, Math.floor(fy * input.h)));
    pipeline.showScan(pipeline.cells.input[0], x, y, input);
    closeup.moveTo(x, y);
  },
  onLeave() {
    if (!player.scanning) pipeline.hideScans('input');
  },
  onPickFilter: i => closeup.setFilter(i),
});
const dense = new DenseView($<HTMLCanvasElement>('denseCanvas'));
const player = new Player(pipeline, closeup, dense);
const history = new History($('history'), $('historyList'), config.historyLimit, entry => void openEntry(entry));

/* ---------- анализ ---------- */

let seq = 0;
let lastRun: { acts: Activations; pred: Prediction } | null = null;

async function decode(url: string): Promise<HTMLImageElement> {
  const img = new Image();
  img.src = url;
  await img.decode();
  return img;
}

async function acceptImage(blob: Blob, source: string): Promise<void> {
  const type = blob.type;
  if (type && !type.startsWith('image/') && type !== 'application/octet-stream') return say('Это не изображение', 'error');
  if (blob.size > config.maxUploadBytes) return say(`Файл больше ${config.maxUploadBytes / 1024 / 1024} МБ`, 'error');

  const url = URL.createObjectURL(blob);
  let img: HTMLImageElement;
  try {
    img = await decode(url);
  } catch {
    URL.revokeObjectURL(url);
    return say('Браузер не смог открыть этот формат. Попробуйте JPG или PNG.', 'error');
  }
  stopCamera();
  const entry = history.add(blob, url, source);
  say(`${source} · ${img.naturalWidth}×${img.naturalHeight}`);
  await analyze(entry, img);
}

async function openEntry(entry: HistoryEntry): Promise<void> {
  stopCamera();
  history.setCurrent(entry.id);
  say(entry.source);
  await analyze(entry, await decode(entry.url));
}

async function analyze(entry: HistoryEntry, img: HTMLImageElement): Promise<void> {
  const run = ++seq;
  frame.showImage(entry.url);
  verdict.pending();
  btnReplay.disabled = btnSkip.disabled = true;

  const local = computeFromImage(img);
  closeup.setInput(local.input);
  player.prepare(local);

  let pred: Prediction;
  frame.root.classList.add('analyzing');
  try {
    pred = await client.predict(entry.blob, local.input);
  } catch (e) {
    if (run !== seq) return;
    verdict.error((e as Error).message);
    return;
  } finally {
    if (run === seq) {
      frame.root.classList.remove('analyzing');
      renderStatus(client.backend);
    }
  }
  if (run !== seq) return;

  history.setResult(entry.id, pred.label, Math.max(pred.probabilities.cat, pred.probabilities.dog));
  const acts = mergeActivations(local, pred.activations);
  lastRun = { acts, pred };
  await play(run);
}

async function play(run: number): Promise<void> {
  if (!lastRun) return;
  btnSkip.disabled = false;
  btnReplay.disabled = true;
  const finished = await player.run(lastRun.acts, lastRun.pred);
  if (run !== seq) return;
  if (finished) verdict.show(lastRun.pred);
  btnSkip.disabled = true;
  btnReplay.disabled = false;
}

/* ---------- камера ---------- */

async function startCamera(): Promise<void> {
  try {
    await camera.start();
  } catch (e) {
    return say((e as Error).message, 'error');
  }
  frame.showCamera();
  btnCameraLabel.textContent = 'Снять';
  btnCamera.classList.add('recording');
  say('нажмите «Снять» или на кадр · Esc — отмена');
}

function stopCamera(): void {
  if (!camera.active) return;
  camera.stop();
  btnCameraLabel.textContent = 'Камера';
  btnCamera.classList.remove('recording');
  frame.restore();
}

async function snap(): Promise<void> {
  try {
    await acceptImage(await camera.snap(), 'с камеры');
  } catch (e) {
    say((e as Error).message, 'error');
  }
}

/* ---------- события ---------- */

const sources = new Sources(
  {
    fileInput: $<HTMLInputElement>('fileInput'),
    fileButton: $<HTMLButtonElement>('btnFile'),
    pasteButton: $<HTMLButtonElement>('btnPaste'),
    urlForm: $<HTMLFormElement>('urlForm'),
    urlInput: $<HTMLInputElement>('urlInput'),
  },
  {
    onImage: (blob, source) => void acceptImage(blob, source),
    say,
    setDragging: on => frame.setDragging(on),
  },
);
sources.bind();

frame.root.addEventListener('click', () => (camera.active ? void snap() : sources.openFilePicker()));
frame.root.addEventListener('keydown', e => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    frame.root.click();
  }
});
btnCamera.addEventListener('click', () => void (camera.active ? snap() : startCamera()));
document.addEventListener('keydown', e => {
  if (e.key === 'Escape' && camera.active) {
    stopCamera();
    say('');
  }
});
btnSkip.addEventListener('click', () => player.skip());
btnReplay.addEventListener('click', () => void play(seq));

onThemeChange(() => player.repaint());
$('themeToggle').addEventListener('click', () => setTheme(effectiveTheme() === 'dark' ? 'light' : 'dark'));

/* ---------- старт ---------- */

$('pasteKey').textContent = PASTE_SHORTCUT;
pipeline.build(null);
pipeline.markFilter(closeup.filterIndex);
dense.draw();
renderStatus(null);
void client.detect().then(renderStatus);
