// Градиентный шум Перлина, 1D, с сидом.
//
// Нужен для микродвижений головы и микродрожи взгляда. Отдельный модуль по
// одной причине: у каждого канала должен быть свой сид, и сиды не должны
// коррелировать. Если два канала возьмут один поток чисел, их периодичности
// совпадут, и на второй минуте наблюдения это станет видно как «дыхание в такт
// повороту головы» — ровно тот эффект, который слой микроповедения и должен
// убирать.

/** xorshift32: детерминированный поток из целого сида. */
function xorshift32(seed) {
  let s = seed | 0 || 1;
  return () => {
    s ^= s << 13; s |= 0;
    s ^= s >>> 17;
    s ^= s << 5; s |= 0;
    return (s >>> 0) / 4294967296;
  };
}

const SIZE = 256;
// Замерено на 20 тыс. отсчётов: fbm(2 октавы) укладывается в ±0.4.
const FBM_NORM = 2.5;
const MASK = SIZE - 1;

export class Noise1D {
  constructor(seed) {
    const rnd = xorshift32(seed);
    // Градиенты в [-1, 1], перестановка индексов — классическая схема Перлина.
    this.g = new Float32Array(SIZE);
    this.p = new Uint8Array(SIZE);
    for (let i = 0; i < SIZE; i++) { this.g[i] = rnd() * 2 - 1; this.p[i] = i; }
    for (let i = SIZE - 1; i > 0; i--) {
      const j = (rnd() * (i + 1)) | 0;
      const t = this.p[i]; this.p[i] = this.p[j]; this.p[j] = t;
    }
  }

  /** Одна октава. Результат примерно в [-1, 1]. */
  at(x) {
    const i0 = Math.floor(x), f = x - i0;
    const a = this.g[this.p[i0 & MASK]];
    const b = this.g[this.p[(i0 + 1) & MASK]];
    // Сглаживающая кривая Перлина 6t^5-15t^4+10t^3: непрерывна по второй
    // производной, поэтому движение не «щёлкает» на границах ячеек.
    const u = f * f * f * (f * (f * 6 - 15) + 10);
    const v0 = a * f, v1 = b * (f - 1);
    return v0 + u * (v1 - v0);
  }

  /**
   * Несколько октав. Две октавы — то, что нужно голове: одна даёт медленный
   * дрейф, вторая чуть заметную неровность. Больше октав превращаются в тремор.
   */
  fbm(x, octaves = 2, lacunarity = 2.3, gain = 0.45) {
    let sum = 0, amp = 1, freq = 1, norm = 0;
    for (let o = 0; o < octaves; o++) {
      sum += this.at(x * freq) * amp;
      norm += amp;
      amp *= gain; freq *= lacunarity;
    }
    // Сырой градиентный шум занимает примерно ±0.4 от диапазона, поэтому
    // растягиваем до ±1: иначе «амплитуда 1.5°» в конфиге означала бы 0.55°,
    // и числа в конфиге пришлось бы подбирать вслепую.
    const v = (sum / norm) * FBM_NORM;
    return v < -1 ? -1 : v > 1 ? 1 : v;
  }
}

/** Логнормальное распределение: интервалы моргания, длинный хвост вправо. */
export class Lognormal {
  constructor(seed, medianSec, sigma) {
    this.rnd = xorshift32(seed);
    this.mu = Math.log(medianSec);
    this.sigma = sigma;
    this._spare = null;
  }

  /** Стандартная нормаль по Боксу — Мюллеру, с кешем второго значения. */
  _normal() {
    if (this._spare !== null) { const v = this._spare; this._spare = null; return v; }
    let u = 0, v = 0;
    while (u === 0) u = this.rnd();
    while (v === 0) v = this.rnd();
    const r = Math.sqrt(-2 * Math.log(u));
    this._spare = r * Math.sin(2 * Math.PI * v);
    return r * Math.cos(2 * Math.PI * v);
  }

  sample() { return Math.exp(this.mu + this.sigma * this._normal()); }
  uniform() { return this.rnd(); }
}
