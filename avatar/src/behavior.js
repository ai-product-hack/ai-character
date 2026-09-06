// Слой микроповедения: взгляд, моргание, движения головы, дыхание.
//
// Работает всегда, поверх всего остального. Без него любое лицо читается как
// труп, и это самый дешёвый эффект в проекте.
//
// Два архитектурных решения, которые дороже всего менять потом:
//
// 1. Взгляд — это точка в МИРОВЫХ координатах, а не локальный поворот костей
//    глаз. При качании головы от шума Перлина глаза тогда контрвращаются сами:
//    получается вестибулоокулярный рефлекс, точка взгляда стоит на месте.
//    Иначе взгляд поплывёт вместе с головой, и лицо будет смотреть «сквозь»
//    собеседника — зритель не опознает причину, но неправильность почувствует.
// 2. У каждого канала свой сид. Общий поток чисел синхронизировал бы
//    периодичности каналов, и на второй минуте это стало бы видно.
//
// Аллокаций в update() нет: все векторы и кватернионы созданы в конструкторе.

import * as THREE from 'three';
import { LAYERS } from './zones.js';
import { Lognormal, Noise1D } from './noise.js';

const DEG = Math.PI / 180;
const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
const smooth = (t) => t * t * (3 - 2 * t);

/** Каналы, которые можно замораживать по одному в отладочной панели. */
export const CHANNELS = Object.freeze(['gaze', 'blink', 'head', 'breath']);

export class Microbehavior {
  /**
   * @param {import('./model.js').AvatarModel} model
   * @param {object} cfg содержимое behavior.config.json
   */
  constructor(model, cfg) {
    this.model = model;
    this.cfg = cfg;
    this.morphs = model.morphs;
    this.enabled = { gaze: true, blink: true, head: true, breath: true };

    const B = model.bones;
    this.bones = {
      neck: B[model.cfg.model.neckBone],
      head: B[model.cfg.model.headBone],
      chest: B[model.cfg.model.chestBone],
      eyeL: B[model.cfg.model.eyeBones[0]],
      eyeR: B[model.cfg.model.eyeBones[1]],
      shoulderL: B[model.cfg.model.shoulderBones[0]],
      shoulderR: B[model.cfg.model.shoulderBones[1]],
    };

    // Позы привязки: всё, что делает этот слой, домножается на них, а не
    // заменяет их. Поза покоя (опущенные руки) уже применена к модели.
    this.rest = new Map();
    for (const [k, bone] of Object.entries(this.bones)) {
      if (bone) this.rest.set(k, bone.quaternion.clone());
    }

    // --- слоты морфов резолвятся один раз ---
    const slot = (n) => this.morphs.slotOf(n);
    this.slots = {
      blinkL: slot('eyeBlinkLeft'), blinkR: slot('eyeBlinkRight'),
      browDL: slot('browDownLeft'), browDR: slot('browDownRight'),
      lookDownL: slot('eyeLookDownLeft'), lookDownR: slot('eyeLookDownRight'),
      lookUpL: slot('eyeLookUpLeft'), lookUpR: slot('eyeLookUpRight'),
    };

    // --- источники случайности, по одному на канал ---
    this.headNoise = {
      yaw: new Noise1D(cfg.head.seed),
      pitch: new Noise1D(cfg.head.seed ^ 0x9e3779b9),
      roll: new Noise1D(cfg.head.seed ^ 0x85ebca6b),
    };
    this.tremorNoise = {
      yaw: new Noise1D(cfg.gaze.seed ^ 0xc2b2ae35),
      pitch: new Noise1D(cfg.gaze.seed ^ 0x27d4eb2f),
    };
    this.blinkRnd = new Lognormal(cfg.blink.seed, cfg.blink.medianSec, cfg.blink.sigma);
    this.gazeRnd = new Lognormal(cfg.gaze.seed, 1, 1);      // используется только uniform()
    this.breathRnd = new Lognormal(cfg.breath.seed, 1, 1);

    // --- состояние взгляда ---
    this.gaze = {
      yaw: 0, pitch: 0,               // текущее смещение от точки на собеседнике, град
      fromYaw: 0, fromPitch: 0,
      toYaw: 0, toPitch: 0,
      phase: 'fixate',                // fixate | saccade
      t: 0, duration: this._fixationDuration(),
    };

    // Линия задержки: голова догоняет взгляд. Кольцевой буфер фиксированного
    // размера — чтобы не аллоцировать в кадре.
    const delayFrames = Math.max(2, Math.ceil(cfg.head.gazeFollow.delayMs / 1000 * 90));
    this.delay = { yaw: new Float32Array(delayFrames), pitch: new Float32Array(delayFrames), i: 0, n: delayFrames };
    this.headFollow = { yaw: 0, pitch: 0 };

    // --- состояние моргания ---
    this.blink = { value: 0, t: 0, phase: 'idle', next: this.blinkRnd.sample(),
                   pendingDouble: false, sinceLast: 99 };

    // --- дыхание ---
    this.breath = { phase: this.breathRnd.uniform(), period: 1 / cfg.breath.hz };
    this._newBreathPeriod();

    this.time = 0;

    // Точки, через которые автомат состояний и эмоция модулируют этот слой.
    // Писать в чужие морфы им нельзя — таблица зон не даст, — поэтому они
    // СМЕЩАЮТ поведение, а не подменяют его.
    this.gazeBias = { yaw: 0, pitch: 0 };       // куда смещены точки фиксации
    this._aversionIn = Infinity;
    this.gazeStyle = null;                      // профиль состояния: контакт/отводы
    this.gazeAnchorIn = Infinity;               // обязательный возврат к собеседнику
    this.blinkScale = 1;                        // множитель интервала моргания
    this.headTiltDeg = 0;                       // текущий наклон от состояния
    this.headPitchDeg = 0;                      // текущий наклон от эмоции
    this.headPoseTarget = { tilt: 0, pitch: 0 };
    this.headPoseSmoothMs = 200;
    this._nod = null;                           // текущий микрокивок
    this._headDelayAcc = 0;

    // --- переиспользуемые объекты ---
    this._v = new THREE.Vector3();
    this._v2 = new THREE.Vector3();
    this._eyePos = new THREE.Vector3();
    this._anchor = new THREE.Vector3();
    this._right = new THREE.Vector3();
    this._up = new THREE.Vector3();
    this._fwd = new THREE.Vector3();
    this._q = new THREE.Quaternion();
    this._e = new THREE.Euler();
    this._target = new THREE.Vector3();

    // Куда смотрит персонаж по умолчанию — мировая точка на собеседнике.
    this.anchor = new THREE.Vector3(0, 1.69, 1);
  }

  /** Точка, на которую направлен взгляд по умолчанию (позиция камеры). */
  setAnchor(v) { this.anchor.copy(v); }

  setEnabled(channel, on) { this.enabled[channel] = !!on; }

  /** Смещение точек фиксации взгляда, в градусах. Задаёт автомат состояний. */
  setGazeBias(yawDeg, pitchDeg) {
    this.gazeBias.yaw = yawDeg || 0;
    this.gazeBias.pitch = pitchDeg || 0;
  }

  /** Переопределения характера саккад для текущего состояния. */
  setGazeStyle(style) {
    const next = style || null;
    if (next === this.gazeStyle) return;
    this.gazeStyle = next;
    this.gaze.averted = false;
    this._aversionIn = this._range(next?.aversion?.everySec, Infinity);
    this.gazeAnchorIn = this._nextAnchorDelay();
  }

  /** Множитель интервала моргания: больше единицы — моргания реже. */
  setBlinkScale(k) { this.blinkScale = k > 0 ? k : 1; }

  /** Постоянные наклоны головы: от состояния (крен) и от эмоции (тангаж). */
  setHeadPose(tiltDeg, pitchDeg, transitionMs = 200) {
    this.headPoseTarget.tilt = tiltDeg || 0;
    this.headPoseTarget.pitch = pitchDeg || 0;
    this.headPoseSmoothMs = Math.max(0, transitionMs);
  }

  /**
   * Резко вернуть взгляд на собеседника. Нужно состоянию `interrupted`:
   * доводить текущую саккаду там нельзя, реакция должна быть мгновенной.
   */
  snapGaze() {
    const g = this.gaze;
    g.phase = 'fixate';
    g.averted = false;
    this._aversionIn = this._range(this.gazeStyle?.aversion?.everySec, Infinity);
    g.t = 0;
    g.duration = this._fixationDuration();
    g.yaw = 0; g.pitch = 0;
    g.fromYaw = 0; g.fromPitch = 0;
    g.toYaw = 0; g.toPitch = 0;
  }

  /** Одиночный микрокивок. Складывается с шумом головы. */
  nod(amplitudeDeg, durationMs) {
    this.gesture({ pitchDeg: amplitudeDeg, durationMs });
  }

  gesture({ pitchDeg = 0, yawDeg = 0, rollDeg = 0, durationMs = 650 }) {
    // Do not restart an unfinished gesture on every streamed clause.
    if (this._nod) return;
    this._nod = { t: 0, amp: pitchDeg, yaw: yawDeg, roll: rollDeg, dur: durationMs / 1000 };
  }

  /**
   * Внешнее событие, к которому стоит привязать моргание: конец фразы,
   * смена состояния. Люди моргают на границах фраз, и это сильный сигнал.
   */
  notifyEvent() {
    // Граница фразы — сильный человеческий сигнал, здесь моргание уместно
    // даже вне очереди, но не чаще рефрактерного промежутка.
    if (this.blink.phase === 'idle' && this.blink.sinceLast >= this.cfg.blink.refractorySec &&
        this.gazeRnd.uniform() < this.cfg.blink.onEventChance) {
      this._startBlink();
      this.blink.next = this.blinkRnd.sample() * this.blinkScale;
    }
  }

  // ---------------------------------------------------------------- взгляд

  _fixationDuration() {
    const f = this.cfg.gaze.fixation;
    const scale = this.gazeStyle?.fixationScale ?? 1;
    return (f.minMs + this.gazeRnd.uniform() * (f.maxMs - f.minMs))
      * scale / 1000;
  }

  _nextAnchorDelay() {
    const range = this.gazeStyle?.anchorEverySec;
    if (!range) return Infinity;
    return range[0] + this.gazeRnd.uniform() * (range[1] - range[0]);
  }

  _range(range, fallback = 0) {
    return range ? range[0] + this.gazeRnd.uniform() * (range[1] - range[0]) : fallback;
  }

  _startSaccade(yaw, pitch, holdSec = null) {
    const S = this.cfg.gaze.saccade, L = this.cfg.gaze.limitDeg;
    const P = this.gazeStyle || {};
    const g = this.gaze;
    const big = this.gazeRnd.uniform() < (P.largeChance ?? S.largeChance);
    const [lo, hi] = big ? S.largeAmplitudeDeg : S.amplitudeDeg;
    const dir = this.gazeRnd.uniform() * Math.PI * 2;
    const amplitudeScale = P.amplitudeScale ?? 1;

    g.fromYaw = g.yaw; g.fromPitch = g.pitch;

    // Точка фиксации выбирается ОТ ЯКОРЯ, а не от текущего положения. Смещение
    // от текущего — это случайное блуждание: за десяток саккад взгляд уходит
    // на предел поворота и там остаётся, что выглядит как «уставился в стену».
    // Человек же разглядывает лицо собеседника, всё время возвращаясь к нему.
    // Все цели отсчитываются от смещения, заданного состоянием: в thinking
    // взгляд уходит вверх-влево и разглядывает уже ТУ область, а не мечется
    // между ней и собеседником.
    const bx = this.gazeBias.yaw, by = this.gazeBias.pitch;
    // При размышлении человек не держит одну точку в потолке: иногда быстро
    // проверяет лицо собеседника и снова отводит глаза. anchorChance задаёт
    // именно такие короткие возвраты, независимо от смещённого gazeBias.
    const anchorDue = this.gazeAnchorIn <= 0;
    if (anchorDue || ((P.anchorChance ?? 0) > 0 &&
        this.gazeRnd.uniform() < P.anchorChance)) {
      const jitter = P.anchorJitterDeg ?? 0.6;
      g.toYaw = (this.gazeRnd.uniform() * 2 - 1) * jitter;
      g.toPitch = (this.gazeRnd.uniform() * 2 - 1) * jitter;
      this.gazeAnchorIn = this._nextAnchorDelay();
    } else if (this.gazeRnd.uniform() < (P.returnChance ?? S.returnChance)) {
      const jitter = P.returnJitterDeg ?? S.returnJitterDeg;
      g.toYaw = bx + (this.gazeRnd.uniform() * 2 - 1) * jitter;
      g.toPitch = by + (this.gazeRnd.uniform() * 2 - 1) * jitter;
    } else {
      const amp = (lo + this.gazeRnd.uniform() * (hi - lo)) * amplitudeScale;
      g.toYaw = bx + Math.cos(dir) * amp;
      g.toPitch = by + Math.sin(dir) * amp * 0.65;
    }
    if (Number.isFinite(yaw) && Number.isFinite(pitch)) {
      g.toYaw = yaw; g.toPitch = pitch;
    }
    g.holdSec = holdSec;
    g.toYaw = clamp(g.toYaw, -L.yaw, L.yaw);
    g.toPitch = clamp(g.toPitch, -L.pitch, L.pitch);

    // Длительность слабо зависит от амплитуды на малых углах — поэтому база
    // плюс небольшая добавка, а не пропорция.
    const realAmp = Math.hypot(g.toYaw - g.fromYaw, g.toPitch - g.fromPitch);
    g.duration = clamp(S.minMs + realAmp * S.msPerDeg, S.minMs, S.maxMs) / 1000;
    g.phase = 'saccade';
    g.t = 0;

    // Моргание группируется с саккадами — но НЕ добавляется к фоновому потоку,
    // а притягивается к саккаде. Замерено на 90 с: добавление давало 43
    // моргания в минуту при медианном промежутке 1.1 с, то есть человека с
    // нервным тиком. Правильная модель — сдвиг уже назначенного моргания:
    // если оно и так скоро, отыграть его сейчас, на саккаде. Частота остаётся
    // фоновой, а привязка к событию появляется.
    if (this.enabled.blink && this.blink.phase === 'idle' &&
        this.blink.next < this.cfg.blink.alignWindowSec &&
        this.gazeRnd.uniform() < this.cfg.blink.onSaccadeChance) {
      this._startBlink();
      this.blink.next = this.blinkRnd.sample() * this.blinkScale;
    }
  }

  _updateGaze(dt) {
    const g = this.gaze;
    this.gazeAnchorIn -= dt;
    if (!g.averted) this._aversionIn -= dt;
    g.t += dt;
    if (g.phase === 'fixate') {
      const aversion = this.gazeStyle?.aversion;
      if (g.averted) {
        // Hold one off-face target, then return. Do not scatter random targets
        // around the room or let the contact timer interrupt the glance.
        if (g.t >= g.duration) {
          g.averted = false;
          this._aversionIn = this._range(aversion?.everySec, Infinity);
          this.gazeAnchorIn = this._nextAnchorDelay();
          this._startSaccade(0, 0);
        }
      } else if (aversion && this._aversionIn <= 0) {
        g.averted = true;
        const side = this.gazeRnd.uniform() < 0.5 ? -1 : 1;
        this._startSaccade(side * this._range(aversion.yawDeg),
          this._range(aversion.pitchDeg), this._range(aversion.holdSec));
      } else if (g.t >= g.duration || this.gazeAnchorIn <= 0) this._startSaccade();
    } else {
      if (g.t >= g.duration) {
        g.yaw = g.toYaw; g.pitch = g.toPitch;
        g.phase = 'fixate'; g.t = 0; g.duration = g.holdSec ?? this._fixationDuration();
      } else {
        // Бросок резкий. Здесь намеренно НЕ плавная интерполяция за сотни
        // миллисекунд: настоящий глаз фиксируется и прыгает.
        const k = smooth(g.t / g.duration);
        g.yaw = g.fromYaw + (g.toYaw - g.fromYaw) * k;
        g.pitch = g.fromPitch + (g.toPitch - g.fromPitch) * k;
      }
    }
  }

  /** Куда смотреть: мировая точка, вычисленная от якоря. */
  _gazePoint(out) {
    const F = this.cfg.gaze.fixation;
    const eyeL = this.bones.eyeL;
    eyeL.getWorldPosition(this._eyePos);

    // Базис вокруг направления «глаз -> собеседник».
    this._fwd.copy(this.anchor).sub(this._eyePos);
    const dist = this._fwd.length() || 1;
    this._fwd.divideScalar(dist);
    this._right.set(0, 1, 0).cross(this._fwd).normalize();
    this._up.copy(this._fwd).cross(this._right).normalize();

    // Микродрожь во время фиксации: ровно неподвижный глаз читается как
    // стоп-кадр, но амплитуда здесь — десятые доли градуса, не больше.
    const trem = this.gaze.phase === 'fixate' ? F.tremorDeg : 0;
    const yaw = this.gaze.yaw + trem * this.tremorNoise.yaw.at(this.time * F.tremorHz);
    const pitch = this.gaze.pitch + trem * this.tremorNoise.pitch.at(this.time * F.tremorHz);

    return out.copy(this.anchor)
      .addScaledVector(this._right, Math.tan(yaw * DEG) * dist)
      .addScaledVector(this._up, Math.tan(pitch * DEG) * dist);
  }

  /**
   * Навести кость глаза на мировую точку. Локальная +Z у костей глаз этой
   * модели смотрит вперёд (проверено), поэтому годится штатный lookAt, который
   * разворачивает +Z на цель и сам учитывает мировую матрицу родителя —
   * отсюда и берётся компенсация движения головы.
   */
  _aimEye(bone, point) {
    if (!bone) return;
    bone.lookAt(point);
    bone.updateMatrix();
  }

  // -------------------------------------------------------------- моргание

  _startBlink(isSecond = false) {
    const B = this.cfg.blink;
    this.blink.phase = 'close';
    this.blink.t = 0;
    if (!isSecond) this.blink.sinceLast = 0;
    this.blink.pendingDouble = !isSecond && this.blinkRnd.uniform() < B.doubleChance;
  }

  _updateBlink(dt) {
    const B = this.cfg.blink, b = this.blink;
    b.sinceLast += dt;
    if (b.phase === 'idle') {
      b.next -= dt;
      // Рефрактерный промежуток: два моргания подряд физически невозможны.
      if (b.next <= 0 && b.sinceLast >= B.refractorySec) {
        this._startBlink();
        b.next = this.blinkRnd.sample() * this.blinkScale;
      }
      b.value = 0;
      return;
    }
    b.t += dt * 1000;
    // Асимметрия по времени: закрытие быстрее открытия. Симметричное моргание
    // читается как затвор фотоаппарата.
    if (b.phase === 'close') {
      b.value = smooth(clamp(b.t / B.closeMs, 0, 1));
      if (b.t >= B.closeMs) { b.phase = 'hold'; b.t = 0; }
    } else if (b.phase === 'hold') {
      b.value = 1;
      if (b.t >= B.holdMs) { b.phase = 'open'; b.t = 0; }
    } else if (b.phase === 'open') {
      b.value = 1 - smooth(clamp(b.t / B.openMs, 0, 1));
      if (b.t >= B.openMs) {
        if (b.pendingDouble) { b.phase = 'gap'; b.t = 0; }
        else { b.phase = 'idle'; b.value = 0; }
      }
    } else if (b.phase === 'gap') {
      b.value = 0;
      if (b.t >= B.doubleGapMs) this._startBlink(true);
    }
  }

  // ------------------------------------------------------- голова и дыхание

  _updateHead(dt) {
    const H = this.cfg.head;
    const G = H.gazeFollow;

    // Морфы состояния натекают через EmotionLayer, а наклон головы раньше
    // применялся мгновенно. На speaking -> listening это был скачок 3.5° за
    // один кадр — лицо выглядело плавным, но весь силуэт заметно дёргался.
    const poseK = this.headPoseSmoothMs > 0
      ? 1 - Math.exp(-dt * 1000 / this.headPoseSmoothMs) : 1;
    this.headTiltDeg += (this.headPoseTarget.tilt - this.headTiltDeg) * poseK;
    this.headPitchDeg += (this.headPoseTarget.pitch - this.headPitchDeg) * poseK;

    // Линия задержки: голова доворачивает туда же, куда ушёл взгляд, но позже
    // и на меньший угол.
    const d = this.delay;
    this._headDelayAcc += dt;
    const step = Math.max(0.001, G.delayMs / 1000 / d.n);
    while (this._headDelayAcc >= step) {
      d.yaw[d.i] = this.gaze.yaw; d.pitch[d.i] = this.gaze.pitch;
      d.i = (d.i + 1) % d.n;
      this._headDelayAcc -= step;
    }
    const delayedYaw = d.yaw[d.i], delayedPitch = d.pitch[d.i];

    const k = G.enabled ? clamp(dt / (G.smoothMs / 1000), 0, 1) : 1;
    this.headFollow.yaw += (delayedYaw * G.gain - this.headFollow.yaw) * k;
    this.headFollow.pitch += (delayedPitch * G.gain - this.headFollow.pitch) * k;

    const t = this.time * H.hz;
    const A = H.amplitudeDeg;
    const nYaw = this.headNoise.yaw.fbm(t, H.octaves) * A.yaw;
    const nPitch = this.headNoise.pitch.fbm(t + 11.3, H.octaves) * A.pitch;
    let nRoll = this.headNoise.roll.fbm(t + 27.7, H.octaves) * A.roll;

    // Микрокивок: короткий импульс поверх шума, а не отдельный канал.
    let nodPitch = 0, gestureYaw = 0, gestureRoll = 0;
    if (this._nod) {
      this._nod.t += dt;
      const k = this._nod.t / this._nod.dur;
      if (k >= 1) this._nod = null;
      else {
        const envelope = Math.sin(k * Math.PI) ** 2;
        nodPitch = envelope * this._nod.amp;
        gestureYaw = Math.sin(k * Math.PI * 2) * envelope * this._nod.yaw;
        gestureRoll = envelope * this._nod.roll;
      }
    }

    const follow = G.enabled ? this.headFollow : { yaw: 0, pitch: 0 };
    const totalYaw = nYaw + follow.yaw + gestureYaw;
    // Gaze pitch and expression pitch use positive-up degrees; local Euler X
    // uses positive-down on this rig. Nods already use positive-down.
    const totalPitch = nPitch - follow.pitch + nodPitch - this.headPitchDeg;
    nRoll += this.headTiltDeg + gestureRoll;

    // Движение делится между шеей и головой: одна кость на всё выглядит как
    // поворот манекена.
    const share = H.neckShare;
    this._applyEuler('neck', totalPitch * share * DEG, totalYaw * share * DEG, nRoll * share * DEG);
    this._applyEuler('head', totalPitch * (1 - share) * DEG, totalYaw * (1 - share) * DEG,
      nRoll * (1 - share) * DEG);
  }

  _newBreathPeriod() {
    const B = this.cfg.breath;
    this.breath.period = (1 / B.hz) * (1 + (this.breathRnd.uniform() * 2 - 1) * B.periodJitter);
  }

  /** Вдох быстрее выдоха: 40 на 60. Симметричная синусоида читается как насос. */
  _breathCurve(phase) {
    const f = this.cfg.breath.inhaleFraction;
    return phase < f
      ? smooth(phase / f)
      : 1 - smooth((phase - f) / (1 - f));
  }

  _updateBreath(dt) {
    const B = this.cfg.breath, br = this.breath;
    br.phase += dt / br.period;
    while (br.phase >= 1) { br.phase -= 1; this._newBreathPeriod(); }

    const v = this._breathCurve(br.phase);
    const lagged = this._breathCurve((br.phase - B.shoulderLagFraction + 1) % 1);

    this._applyEuler('chest', -v * B.chestDeg * DEG, 0, 0);
    // Плечи идут чуть позже груди и расходятся в стороны, а не вверх.
    this._applyEuler('shoulderL', 0, 0, -lagged * B.shoulderDeg * DEG);
    this._applyEuler('shoulderR', 0, 0, lagged * B.shoulderDeg * DEG);
  }

  /** Повернуть кость относительно её позы привязки. */
  _applyEuler(key, x, y, z) {
    const bone = this.bones[key];
    if (!bone) return;
    const rest = this.rest.get(key);
    this._e.set(x, y, z);
    bone.quaternion.copy(rest).multiply(this._q.setFromEuler(this._e));
  }

  // ------------------------------------------------------------------ кадр

  /**
   * Один кадр. Вызывается до commit() writer'а морфов — этот слой только
   * вносит вклады, раскладывает их по мешам writer.
   */
  update(dt, timeSec) {
    this.time = timeSec;
    const S = this.slots;
    const m = this.morphs;

    if (this.enabled.head) this._updateHead(dt);
    if (this.enabled.breath) this._updateBreath(dt);

    if (this.enabled.gaze) {
      this._updateGaze(dt);
      // Кости выше по цепочке уже повёрнуты в этом кадре, поэтому мировые
      // матрицы надо пересчитать до наведения глаз — иначе компенсация
      // движения головы отстанет ровно на кадр.
      this.model.root.updateMatrixWorld(true);
      this._gazePoint(this._target);
      this._aimEye(this.bones.eyeL, this._target);
      this._aimEye(this.bones.eyeR, this._target);

      // Веко следует за взглядом: вниз — опускается, вверх — приподнимается.
      const L = this.cfg.gaze.limitDeg;
      const p = clamp(this.gaze.pitch / L.pitch, -1, 1) * this.cfg.gaze.lidFollow;
      if (p < 0) {
        m.writeSlot(LAYERS.IDLE, S.lookDownL, -p);
        m.writeSlot(LAYERS.IDLE, S.lookDownR, -p);
      } else if (p > 0) {
        m.writeSlot(LAYERS.IDLE, S.lookUpL, p);
        m.writeSlot(LAYERS.IDLE, S.lookUpR, p);
      }
    }

    if (this.enabled.blink) {
      this._updateBlink(dt);
      const v = this.blink.value;
      if (v > 0) {
        m.writeSlot(LAYERS.IDLE, S.blinkL, v);
        m.writeSlot(LAYERS.IDLE, S.blinkR, v);
        // Микроопускание бровей: складывается с эмоцией, не подменяет её.
        const dip = v * this.cfg.blink.browDip;
        m.writeSlot(LAYERS.IDLE, S.browDL, dip);
        m.writeSlot(LAYERS.IDLE, S.browDR, dip);
      }
    }
  }

  /** Состояние каналов для оверлея. */
  debug() {
    return {
      gazePhase: this.gaze.phase,
      averted: !!this.gaze.averted,
      gazeYaw: this.gaze.yaw, gazePitch: this.gaze.pitch,
      blinkPhase: this.blink.phase, blinkValue: this.blink.value,
      nextBlinkSec: this.blink.next,
      breathPhase: this.breath.phase, breathPeriod: this.breath.period,
      headFollowYaw: this.headFollow.yaw,
    };
  }
}
