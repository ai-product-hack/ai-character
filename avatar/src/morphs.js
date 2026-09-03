// Writer морф-таргетов: одна точка записи весов на всю модель.
//
// Зачем отдельная подсистема, а не пара строк в рендер-лупе. В этой модели рот
// распределён по трём мешам (Head_Mesh, Teeth_Mesh, Tongue_Mesh), брови — по
// двум (EyeAO_Mesh, Eyelash_Mesh), моргание — по трём. Порядок морфов в мешах
// разный: jawOpen — индекс 49 в голове и 16 в зубах. Запись «по индексу головы»
// даёт классический баг «губы шевелятся, зубы стоят»; запись в один меш даёт
// ресницы, отставшие от бровей. Поэтому адресация только по имени и только
// через этот класс.
//
// Стоимость кадра: имена резолвятся в слоты один раз при построении, слои
// пишут в Float32Array, commit() проходит по плоским типизированным массивам.
// Аллокаций в рендер-лупе нет.

import { LAYERS, OWNER, UNUSED_MORPHS, ZONE_MORPHS } from './zones.js';

const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);

export class MorphWriter {
  /**
   * @param {Array<{name:string, morphTargetDictionary:Object, morphTargetInfluences:Array<number>}>} meshes
   *        Меши с морфами. Форма совпадает с THREE.Mesh, так что тесты
   *        обходятся простыми объектами — three.js для них не нужен.
   * @param {{strict?:boolean}} [opts] strict: бросать на нарушении владения
   *        вместо счётчика. Включать в тестах, не в проде.
   */
  constructor(meshes, opts = {}) {
    this.strict = !!opts.strict;

    /** @type {Map<string, number>} имя морфа -> слот */
    this.slots = new Map();
    /** @type {string[]} слот -> имя */
    this.names = [];
    /** слот -> плоские пары (массив весов меша, индекс в нём) */
    this._targetArrays = [];   // Array<Array<Float64Array|number[]>>
    this._targetIndices = [];  // Array<Int32Array>
    /** слот -> имя слоя-владельца (или null, если морфа нет в таблице зон) */
    this._owner = [];
    /** слот -> список имён мешей, где морф объявлен. Для диагностики и тестов. */
    this.meshesOf = new Map();

    const collected = new Map(); // имя -> [{influences, index, mesh}]
    for (const mesh of meshes) {
      const dict = mesh.morphTargetDictionary;
      if (!dict || !mesh.morphTargetInfluences) continue;
      for (const [name, index] of Object.entries(dict)) {
        if (!collected.has(name)) collected.set(name, []);
        collected.get(name).push({
          influences: mesh.morphTargetInfluences,
          index,
          mesh: mesh.name || '?',
        });
      }
    }

    for (const [name, entries] of collected) {
      const slot = this.names.length;
      this.slots.set(name, slot);
      this.names.push(name);
      this._targetArrays.push(entries.map((e) => e.influences));
      this._targetIndices.push(Int32Array.from(entries.map((e) => e.index)));
      this._owner.push(OWNER.get(name) || null);
      this.meshesOf.set(name, entries.map((e) => e.mesh));
    }

    this.values = new Float32Array(this.names.length);

    // Диагностика для оверлея: писать в морф, которого в модели нет, или в
    // чужую зону — это ошибка кода, а не состояние сцены. Считаем и показываем.
    this.stats = { unknownWrites: 0, trespassWrites: 0, lastUnknown: null, lastTrespass: null };

    // Морфы модели, которых нет ни в одной зоне и ни в списке неиспользуемых:
    // такое означает, что модель разошлась с таблицей зон.
    this.unclaimed = this.names.filter(
      (n) => !OWNER.has(n) && !UNUSED_MORPHS.includes(n));
    // И наоборот: зоны ссылаются на морф, которого в модели нет.
    this.missing = [];
    for (const names of Object.values(ZONE_MORPHS)) {
      for (const n of names) if (!this.slots.has(n)) this.missing.push(n);
    }
  }

  /** Слот морфа для горячего пути: слой резолвит имя один раз, дальше пишет по слоту. */
  slotOf(name) {
    const s = this.slots.get(name);
    return s === undefined ? -1 : s;
  }

  /** Обнулить кадр. Вызывается один раз в начале кадра, до работы слоёв. */
  begin() {
    this.values.fill(0);
  }

  /**
   * Записать вес. Владение проверяется здесь же, на записи: чужой слой
   * отбрасывается и считается, а не переписывает владельца «последним словом».
   * @returns {boolean} была ли запись принята
   */
  set(layer, name, weight) {
    const slot = this.slots.get(name);
    if (slot === undefined) {
      this.stats.unknownWrites++;
      this.stats.lastUnknown = name;
      if (this.strict) throw new Error(`MorphWriter: морфа «${name}» нет в модели`);
      return false;
    }
    return this.setSlot(layer, slot, weight);
  }

  /** То же по слоту — без хеширования строки, для рендер-лупа. */
  setSlot(layer, slot, weight) {
    if (this._owner[slot] !== layer) {
      this.stats.trespassWrites++;
      this.stats.lastTrespass = `${layer} -> ${this.names[slot]} (владелец: ${this._owner[slot] || 'никто'})`;
      if (this.strict) {
        throw new Error(
          `MorphWriter: слой «${layer}» пишет в «${this.names[slot]}», ` +
          `которым владеет «${this._owner[slot] || 'никто'}»`);
      }
      return false;
    }
    this.values[slot] = weight;
    return true;
  }

  /** Прибавить к уже записанному в этом кадре. Сумма клампится в commit(). */
  add(layer, name, weight) {
    const slot = this.slots.get(name);
    if (slot === undefined) {
      this.stats.unknownWrites++;
      this.stats.lastUnknown = name;
      if (this.strict) throw new Error(`MorphWriter: морфа «${name}» нет в модели`);
      return false;
    }
    return this.addSlot(layer, slot, weight);
  }

  addSlot(layer, slot, weight) {
    if (this._owner[slot] !== layer) {
      this.stats.trespassWrites++;
      this.stats.lastTrespass = `${layer} -> ${this.names[slot]} (владелец: ${this._owner[slot] || 'никто'})`;
      if (this.strict) {
        throw new Error(
          `MorphWriter: слой «${layer}» прибавляет к «${this.names[slot]}», ` +
          `которым владеет «${this._owner[slot] || 'никто'}»`);
      }
      return false;
    }
    this.values[slot] += weight;
    return true;
  }

  /**
   * Разложить кадр по всем мешам. Один проход, без аллокаций.
   * Клампинг суммы в [0, 1] здесь: слой может складывать несколько вкладов
   * (висема + дожим), и уводить вес за единицу нельзя.
   */
  commit() {
    const { values, _targetArrays, _targetIndices } = this;
    for (let slot = 0; slot < values.length; slot++) {
      const v = clamp01(values[slot]);
      values[slot] = v;
      const arrays = _targetArrays[slot];
      const indices = _targetIndices[slot];
      for (let i = 0; i < arrays.length; i++) arrays[i][indices[i]] = v;
    }
  }

  /** Текущий разложенный вес морфа. Для оверлея, панели и тестов. */
  get(name) {
    const slot = this.slots.get(name);
    return slot === undefined ? undefined : this.values[slot];
  }

  /** Отчёт о расхождении модели и таблицы зон. Печатается один раз на загрузке. */
  describe() {
    const byLayer = {};
    for (const layer of Object.values(LAYERS)) byLayer[layer] = 0;
    let unowned = 0;
    for (const owner of this._owner) {
      if (owner) byLayer[owner]++; else unowned++;
    }
    return {
      morphs: this.names.length,
      meshes: new Set([...this.meshesOf.values()].flat()).size,
      byLayer,
      unowned,
      unclaimed: this.unclaimed,
      missing: this.missing,
    };
  }
}

/** Собрать writer по загруженной сцене glTF. */
export function writerFromScene(root, opts) {
  const meshes = [];
  root.traverse((o) => {
    if (o.isMesh && o.morphTargetDictionary && o.morphTargetInfluences) meshes.push(o);
  });
  return new MorphWriter(meshes, opts);
}
