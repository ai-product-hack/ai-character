// Глаза: своя роговица и блики.
//
// В экспорте оба глаза — 312 треугольников на двоих, то есть очень грубая
// сфера. Зеркальный блик на такой геометрии выходит гранёным, и никакой
// clearcoat на самом глазном яблоке этого не чинит: гранёные нормали никуда
// не денутся. Поэтому сверху кладётся собственный колпачок роговицы —
// полусфера на 32 сегмента, прозрачная, с низкой шероховатостью. Около сотни
// треугольников на глаз, и появляются влажность и параллакс.
//
// Catchlight — отдельный эмиссивный билборд, привязанный к ключевому свету.
// Блики на двух глазах намеренно не симметричны: симметрия читается как кукла.

import * as THREE from 'three';

/** Радиус глазного яблока и его центр по геометрии меша глаз. */
function eyeballBounds(eyeMesh, eyeBone) {
  const box = new THREE.Box3().setFromObject(eyeMesh);
  const size = box.getSize(new THREE.Vector3());
  // Меш глаз содержит оба яблока: по X там два глаза, радиус берём по Y.
  const radius = size.y * 0.5;
  const center = eyeBone ? eyeBone.getWorldPosition(new THREE.Vector3()) : box.getCenter(new THREE.Vector3());
  return { radius, center };
}

export class Eyes {
  /**
   * @param {THREE.Object3D} root корень загруженной модели
   * @param {object} cfg cfg.eyes из look.config.json
   * @param {{left:THREE.Object3D, right:THREE.Object3D}} bones кости глаз
   */
  constructor(root, cfg, bones) {
    this.cfg = cfg;
    this.bones = bones;
    this.corneas = [];
    this.catchlights = [];

    const eyeMesh = root.getObjectByName('Eye_Mesh');
    if (!eyeMesh) return;
    eyeMesh.updateWorldMatrix(true, true);

    const { radius } = eyeballBounds(eyeMesh, bones.left);
    this.radius = radius;

    for (const side of ['left', 'right']) {
      const bone = bones[side];
      if (!bone) continue;
      if (cfg.cornea.enabled) this.corneas.push(this._makeCornea(bone, side));
      if (cfg.catchlight.enabled) this.catchlights.push(this._makeCatchlight(bone, side));
    }
  }

  /**
   * Колпачок роговицы. Вешается на кость глаза, поэтому едет вместе со взглядом
   * без единой строки в рендер-лупе.
   */
  _makeCornea(bone, side) {
    const c = this.cfg.cornea;
    // Не вся сфера, а колпачок над радужкой: thetaLength ограничивает дугу.
    const geo = new THREE.SphereGeometry(
      this.radius * c.radiusScale, c.segments, Math.max(8, c.segments >> 1),
      0, Math.PI * 2, 0, c.arcDeg * Math.PI / 180);
    // SphereGeometry смотрит колпачком в +Y, глаз — в +Z.
    geo.rotateX(Math.PI / 2);

    // Чёрный базовый цвет + аддитивное смешивание. Диффузной составляющей у
    // роговицы нет физически, и первая версия это подтвердила от противного:
    // белый цвет с opacity 0.28 дал белое бельмо на весь глаз — в HDR даже
    // 28% от ярко освещённой белой поверхности выбивает в единицу. С чёрной
    // базой в кадр попадает ровно то, что и должно: отражение и clearcoat.
    const mat = new THREE.MeshPhysicalMaterial({
      color: 0x000000,
      transparent: true,
      opacity: 1,
      blending: THREE.AdditiveBlending,
      roughness: c.roughness,
      metalness: 0,
      clearcoat: c.clearcoat,
      clearcoatRoughness: c.clearcoatRoughness,
      ior: c.ior,
      envMapIntensity: c.envMapIntensity,
      depthWrite: false,             // прозрачное поверх глаза — не пишем глубину
      side: THREE.FrontSide,
    });

    const mesh = new THREE.Mesh(geo, mat);
    mesh.name = `cornea_${side}`;
    mesh.position.z = c.forwardOffset;
    mesh.renderOrder = 10;           // после глаза, до ресниц
    mesh.castShadow = false;
    bone.add(mesh);
    return mesh;
  }

  /**
   * Блик от ключевого света. Билборд, а не точечный источник: настоящий блик
   * от SpotLight на такой грубой сфере ловится непредсказуемо, а зрителю нужен
   * ровно один чёткий catchlight в предсказуемом месте.
   */
  _makeCatchlight(bone, side) {
    const c = this.cfg.catchlight;
    const asym = side === 'right' ? c.sizeAsymmetry : 1;
    // Размер в долях радиуса яблока, а не в метрах: абсолютное число сломается
    // на любой другой модели.
    const geo = new THREE.CircleGeometry(this.radius * c.sizeRel * asym, 12);
    const mat = new THREE.MeshBasicMaterial({
      color: new THREE.Color(c.color),
      transparent: true,
      opacity: 1,
      depthWrite: false,
      depthTest: false,              // блик всегда поверх роговицы
      toneMapped: false,
    });
    mat.color.multiplyScalar(c.intensity);

    const mesh = new THREE.Mesh(geo, mat);
    mesh.name = `catchlight_${side}`;
    const [ox, oy] = side === 'right' ? c.offsetRight : c.offsetLeft;
    mesh.position.set(
      ox * this.radius,
      oy * this.radius,
      this.radius * this.cfg.cornea.radiusScale * 1.02,
    );
    mesh.renderOrder = 20;
    bone.add(mesh);
    return mesh;
  }

  /** Материал радужки: с роговицей поверх ей нужна другая шероховатость. */
  applyIrisMaterial(root) {
    const eyeMesh = root.getObjectByName('Eye_Mesh');
    if (!eyeMesh) return;
    const mats = Array.isArray(eyeMesh.material) ? eyeMesh.material : [eyeMesh.material];
    for (const m of mats) {
      m.roughness = this.cfg.iris.roughness;
      m.envMapIntensity = this.cfg.iris.envMapIntensity;
      m.needsUpdate = true;
    }
  }

  setVisible(v) {
    for (const m of [...this.corneas, ...this.catchlights]) m.visible = v;
  }
}
