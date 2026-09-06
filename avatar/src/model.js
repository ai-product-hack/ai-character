// Загрузка модели и приведение её к тому, что ждёт рендер.
//
// Всё, что здесь чинится, найдено на инвентаризации (avatar/MODEL_REPORT.md):
// маска на материале головы, BLEND+doubleSided на ресницах и EyeAO, невидимая
// в кадре обувь. Правки профилактические — ставятся сразу, а не по факту багов.

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { writerFromScene } from './morphs.js';
import { Eyes } from './eyes.js';

export class AvatarModel {
  constructor(gltf, cfg) {
    this.gltf = gltf;
    this.root = gltf.scene;
    this.cfg = cfg;

    this.bones = {};
    this.meshes = [];
    this.root.traverse((o) => {
      if (o.isBone || o.isObject3D) this.bones[o.name] = this.bones[o.name] || o;
      if (o.isMesh) this.meshes.push(o);
    });

    this._hideInvisibleMeshes();
    this._applyRestPose();
    this._fixMaterials();

    // Единая точка записи весов на всю модель.
    this.morphs = writerFromScene(this.root);

    const M = cfg.model;
    this.eyes = new Eyes(this.root, cfg.eyes, {
      left: this.bones[M.eyeBones[0]],
      right: this.bones[M.eyeBones[1]],
    });
    this.eyes.applyIrisMaterial(this.root);
  }

  _hideInvisibleMeshes() {
    for (const name of this.cfg.model.hiddenMeshes) {
      const o = this.root.getObjectByName(name);
      if (o) { o.visible = false; o.frustumCulled = true; }
    }
  }

  /**
   * Опустить руки. Модель экспортирована в T-позе, и на кадре по грудь руки
   * входят в кадр по краям. Углы домножаются на bind-поворот, а не заменяют
   * его: поворот кости в её собственных осях зависит от того, как ригер
   * сориентировал bind, и «поставить абсолютный угол» здесь означало бы
   * подбирать числа под чужую систему координат.
   */
  _applyRestPose() {
    const pose = this.cfg.model.restPose || {};
    const q = new THREE.Quaternion();
    const e = new THREE.Euler();
    for (const [name, deg] of Object.entries(pose)) {
      if (name.startsWith('_')) continue;
      const bone = this.bones[name];
      if (!bone) continue;
      e.set(deg[0] * Math.PI / 180, deg[1] * Math.PI / 180, deg[2] * Math.PI / 180);
      bone.quaternion.multiply(q.setFromEuler(e));
    }
    this.root.updateMatrixWorld(true);
  }

  _fixMaterials() {
    const M = this.cfg.model;
    const S = this.cfg.skin;

    for (const mesh of this.meshes) {
      mesh.frustumCulled = false;   // морфы двигают вершины за пределы исходного bbox
      // Карта теней рисуется каждый кадр, поэтому в неё попадает только то,
      // что реально отбрасывает видимую тень. Глаза, ресницы, зубы и язык — нет.
      mesh.castShadow = this.cfg.renderer.shadowCasters.includes(mesh.name);
      mesh.receiveShadow = true;

      const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
      for (const mat of mats) {
        // Кожа и голова: зеркальную составляющую приглушаем, факторы
        // metalness/roughness НЕ трогаем — в glTF они умножаются на MR-текстуру,
        // и металличность кожи там уже фактически нулевая.
        if (mat.name === 'Head' || mat.name === 'Body') {
          mat.specularIntensity = S.specularIntensity;
          mat.envMapIntensity = S.envMapIntensity;
        }

        // Маска на голове: под мягким светом с bloom её ступенчатый край
        // заметен алиасингом. Волосяной кромки на самой голове нет, так что
        // маске там взяться не от чего.
        if (M.headMaterialOpaque && mat.name === 'Head' && mat.alphaTest > 0) {
          mat.alphaTest = 0;
          mat.transparent = false;
          mat.depthWrite = true;
        }

        // BLEND + doubleSided — рецепт артефактов сортировки на тёмном фоне.
        // Правило применяется ТОЛЬКО к области глаза. Первая версия снимала
        // depthWrite у всего прозрачного и задевала avaturn_hair_1, который
        // тоже пришёл BLEND: волосы без записи глубины сортируются сами с
        // собой неверно. Область действия задаётся списком в конфиге.
        if (mat.transparent && M.transparentRenderOrder[mesh.name] !== undefined) {
          mat.depthWrite = false;
          if (mesh.name === 'Eyelash_Mesh') mat.side = THREE.FrontSide;
        }
        mat.needsUpdate = true;
      }

      // Явный порядок отрисовки прозрачного: глаз -> AO -> ресницы.
      const order = M.transparentRenderOrder[mesh.name];
      if (order !== undefined) mesh.renderOrder = order;
    }
  }

  /** Мировая точка, на которую наводится кадр (между глаз). */
  frameTarget(out = new THREE.Vector3()) {
    const C = this.cfg.camera;
    const bone = this.bones[C.targetBone];
    if (!bone) return out.set(0, 1.694, 0.084);
    this.root.updateWorldMatrix(true, true);
    bone.getWorldPosition(out);
    // Derive the eye midpoint from the rig: a signed offset from LeftEye
    // was model-specific and placed both the camera and gaze anchor off-centre.
    if (C.targetBetweenEyes) {
      const [left, right] = this.cfg.model.eyeBones.map(name => this.bones[name]);
      if (left && right) {
        left.getWorldPosition(out);
        out.add(right.getWorldPosition(new THREE.Vector3())).multiplyScalar(0.5);
      }
    }
    return out.add(new THREE.Vector3().fromArray(C.targetOffset));
  }

  /** Треугольники видимой части — для оверлея. */
  visibleTriangles() {
    let n = 0;
    for (const m of this.meshes) {
      if (!m.visible) continue;
      const g = m.geometry;
      n += (g.index ? g.index.count : g.attributes.position.count) / 3;
    }
    return Math.round(n);
  }
}

export function loadAvatarModel(cfg, onProgress) {
  return new Promise((resolve, reject) => {
    new GLTFLoader().load(
      cfg.model.url,
      (gltf) => resolve(new AvatarModel(gltf, cfg)),
      onProgress,
      reject,
    );
  });
}
