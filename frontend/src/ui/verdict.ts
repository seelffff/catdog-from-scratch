import type { Prediction } from '../api/types';

const $ = (id: string) => document.getElementById(id)!;
const pct = (p: number) => `${(p * 100).toFixed(1)}%`;

/** Блок «ответ сети»: слово, уверенность, общая полоса кошка|собака и пояснение. */
export class Verdict {
  private readonly root = $('verdict');
  private readonly word = $('verdictWord');
  private readonly conf = $('confNum');
  private readonly note = $('verdictNote');
  private readonly legend = { cat: $('rowCat'), dog: $('rowDog') };
  private readonly bars = { cat: $('barCat'), dog: $('barDog') };
  private readonly nums = { cat: $('pCat'), dog: $('pDog') };

  pending(): void {
    this.setWord('смотрю…', false);
    this.note.textContent = 'Сеть изучает фото.';
    this.reset();
  }

  error(message: string): void {
    this.setWord('ошибка', false);
    this.note.textContent = message;
    this.reset();
  }

  show(pred: Prediction): void {
    const { cat, dog } = pred.probabilities;
    const winner = pred.label;
    const conf = Math.max(cat, dog);

    this.setWord(winner === 'cat' ? 'кошка' : 'собака', true);
    this.conf.textContent = pct(conf);
    for (const k of ['cat', 'dog'] as const) {
      const p = k === 'cat' ? cat : dog;
      this.bars[k].style.width = pct(p);
      this.bars[k].classList.toggle('win', k === winner);
      this.legend[k].classList.toggle('win', k === winner);
      this.nums[k].textContent = pct(p);
    }

    let note =
      conf < 0.6
        ? 'Модель сомневается: оценки кошки и собаки близки. Проверьте ответ.'
        : conf < 0.8
          ? 'Сеть склоняется к этому ответу, но сомневается.'
          : 'Это оценка среди двух классов; правильность ответа не гарантирована.';
    if (pred.mode === 'mock') note += ' Это демо-ответ: сервер модели не подключён.';
    else if (pred.model === 'dummy') note += ' Ответила заглушка сервера, обученная модель ещё не подключена.';
    this.note.textContent = note;
  }

  private setWord(text: string, final: boolean): void {
    this.root.classList.toggle('pending', !final);
    this.word.textContent = text;
    this.word.classList.remove('reveal');
    if (final) {
      void this.word.offsetWidth; // перезапуск анимации
      this.word.classList.add('reveal');
    }
  }

  private reset(): void {
    this.conf.textContent = '—';
    for (const k of ['cat', 'dog'] as const) {
      this.bars[k].style.width = '0';
      this.bars[k].classList.remove('win');
      this.legend[k].classList.remove('win');
      this.nums[k].textContent = '—';
    }
  }
}
