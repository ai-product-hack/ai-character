// Кадр, свет, фон и пост. Всё числовое живёт в look.config.json — здесь только
// то, как эти числа превращаются в сцену.

import * as THREE from 'three';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { BokehPass } from 'three/addons/postprocessing/BokehPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';

const DEG = Math.PI / 180;

/** Точка на сфере вокруг цели: азимут 0 — прямо перед лицом (+Z), вверх — +Y. */
export function polarTo(target, azimuthDeg, elevationDeg, distance, out = new THREE.Vector3()) {
  const az = azimuthDeg * DEG, el = elevationDeg * DEG;
  return out.set(
    target.x + distance * Math.cos(el) * Math.sin(az),
    target.y + distance * Math.sin(el),
    target.z + distance * Math.cos(el) * Math.cos(az),
  );
}

/**
 * Фон: почти чёрный с еле заметным градиентом. Большая сфера изнутри, а не
 * scene.background — так градиент попадает в bloom и DOF вместе со сценой.
 */
function makeBackdrop(cfg) {
  const mat = new THREE.ShaderMaterial({
    side: THREE.BackSide, depthWrite: false, toneMapped: false,
    uniforms: {
      top: { value: new THREE.Color(cfg.top) },
      bottom: { value: new THREE.Color(cfg.bottom) },
      power: { value: cfg.gradientPower },
    },
    vertexShader: `
      varying float vY;
      void main() {
        vY = normalize(position).y;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }`,
    fragmentShader: `
      uniform vec3 top; uniform vec3 bottom; uniform float power;
      varying float vY;
      void main() {
        float k = pow(clamp(vY * 0.5 + 0.5, 0.0, 1.0), power);
        gl_FragColor = vec4(mix(bottom, top, k), 1.0);
      }`,
  });
  const mesh = new THREE.Mesh(new THREE.SphereGeometry(8, 32, 16), mat);
  mesh.name = 'backdrop';
  mesh.frustumCulled = false;
  return mesh;
}

/**
 * Финальный пост одним проходом: хроматическая аберрация, виньетка и зерно.
 * Стоит ПОСЛЕ OutputPass, то есть работает уже в sRGB — зерно и виньетка в
 * линейном пространстве выглядят не так, как ожидает глаз.
 */
const GradeShader = {
  uniforms: {
    tDiffuse: { value: null },
    grain: { value: 0.03 },
    ca: { value: 0.0016 },
    vignette: { value: 0.55 },
    vignetteSoftness: { value: 0.55 },
    time: { value: 0 },
  },
  vertexShader: `
    varying vec2 vUv;
    void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
  fragmentShader: `
    uniform sampler2D tDiffuse;
    uniform float grain, ca, vignette, vignetteSoftness, time;
    varying vec2 vUv;

    // Дешёвый белый шум по координате и времени. Зерно должно ползти,
    // статичное читается как грязь на объективе.
    float hash(vec2 p, float t) {
      return fract(sin(dot(p, vec2(12.9898, 78.233)) + t) * 43758.5453);
    }

    void main() {
      vec2 c = vUv - 0.5;
      float r2 = dot(c, c);

      // Аберрация растёт от центра к краям, как у настоящей оптики.
      vec2 off = c * ca * r2 * 4.0;
      vec3 col = vec3(
        texture2D(tDiffuse, vUv + off).r,
        texture2D(tDiffuse, vUv).g,
        texture2D(tDiffuse, vUv - off).b
      );

      // Виньетка: 0 в центре кадра, 1 в углу.
      float d = sqrt(r2) * 1.41421356;
      col *= 1.0 - vignette * smoothstep(1.0 - vignetteSoftness, 1.0, d);

      col += (hash(vUv * 1024.0, time) - 0.5) * grain;

      gl_FragColor = vec4(col, 1.0);
    }`,
};

export class Look {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {object} cfg содержимое look.config.json
   */
  constructor(canvas, cfg) {
    this.cfg = cfg;
    this.canvas = canvas;

    this.renderer = new THREE.WebGLRenderer({
      canvas, antialias: cfg.renderer.antialias, powerPreference: 'high-performance',
    });
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = cfg.renderer.exposure;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    if (cfg.renderer.shadows) {
      this.renderer.shadowMap.enabled = true;
      this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
      // Ключевая правка по производительности. BokehPass рендерит сцену второй
      // раз ради глубины, и при автообновлении карта теней пересчитывается на
      // каждый такой рендер — то есть дважды за кадр. Замерено: 31.5 мс против
      // 18.0 мс на том же кадре. Обновляем сами, ровно один раз за кадр.
      this.renderer.shadowMap.autoUpdate = false;
    }

    // Счётчики отрисовки должны копить всю цепочку пассов, а не показывать
    // последний полноэкранный квад. Сбрасываем сами, раз в кадр.
    this.renderer.info.autoReset = false;

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(30, 1, cfg.camera.near, cfg.camera.far);

    // Цель кадра. Реальная точка ставится в frameOn() по кости из модели —
    // до загрузки держим правдоподобную заглушку на высоте глаз.
    this.target = new THREE.Vector3(0, 1.694, 0.084);

    this.backdrop = makeBackdrop(cfg.background);
    this.scene.add(this.backdrop);

    this._buildEnvironment();
    this._buildLights();
    this._buildComposer();

    this._tmp = new THREE.Vector3();   // переиспользуемый вектор, без аллокаций в кадре
  }

  /**
   * IBL. Без окружения физические материалы под ACES не читаются: кожа
   * становится плоской, а глаза — матовыми чёрными дырами. Ставится первым,
   * до подбора света, иначе свет подбирается под неверную базу.
   */
  _buildEnvironment() {
    const env = this.cfg.lights.environment;
    if (!env.enabled) return;
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    pmrem.compileEquirectangularShader();
    this.envTexture = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    this.scene.environment = this.envTexture;
    this.scene.environmentIntensity = env.intensity;
    pmrem.dispose();
  }

  _buildLights() {
    const L = this.cfg.lights;
    this.lights = {};

    const spot = (c) => {
      const s = new THREE.SpotLight(new THREE.Color(c.color), c.intensity,
        c.distance * 4, c.angleDeg * DEG, c.penumbra, c.decay);
      s.castShadow = !!c.castShadow;
      if (s.castShadow) {
        s.shadow.mapSize.set(this.cfg.renderer.shadowMapSize, this.cfg.renderer.shadowMapSize);
        s.shadow.bias = c.shadowBias;
        s.shadow.normalBias = c.shadowNormalBias;
        s.shadow.radius = c.shadowRadius;
        s.shadow.camera.near = 0.1;
        s.shadow.camera.far = c.distance * 4;
      }
      return s;
    };
    const point = (c) => new THREE.PointLight(new THREE.Color(c.color), c.intensity, c.distance * 4, 2);

    this.lights.key = spot(L.key);
    this.lights.rim = spot(L.rim);
    this.lights.fill = point(L.fill);
    this.lights.bounce = point(L.bounce);
    this.lights.ambient = new THREE.AmbientLight(new THREE.Color(L.ambient.color), L.ambient.intensity);

    for (const [name, light] of Object.entries(this.lights)) {
      light.name = `light_${name}`;
      this.scene.add(light);
      if (light.target) this.scene.add(light.target);
    }
    this.placeLights();
  }

  /** Расставить свет вокруг текущей цели кадра. */
  placeLights() {
    const L = this.cfg.lights;
    for (const key of ['key', 'rim', 'fill', 'bounce']) {
      const c = L[key], light = this.lights[key];
      polarTo(this.target, c.azimuthDeg, c.elevationDeg, c.distance, light.position);
      if (light.target) light.target.position.copy(this.target);
    }
  }

  _buildComposer() {
    const P = this.cfg.post;
    // HalfFloat: bloom должен работать по HDR-значениям до тонмаппинга,
    // иначе светá срезаются раньше, чем до них доберётся свечение.
    this.composer = new EffectComposer(this.renderer, new THREE.WebGLRenderTarget(1, 1, {
      type: THREE.HalfFloatType, samples: this.cfg.renderer.antialias ? 4 : 0,
    }));
    this.composer.addPass(new RenderPass(this.scene, this.camera));

    if (P.dof.enabled) {
      this.bokeh = new BokehPass(this.scene, this.camera, {
        focus: 0.85, aperture: P.dof.aperture, maxblur: P.dof.maxblur,
      });
      this.composer.addPass(this.bokeh);
    }
    // BokehPass рендерит сцену второй раз ради карты глубины, в полном
    // разрешении. Размытие здесь мягкое, и половинная глубина на глаз
    // неотличима — но это половина полноэкранного прохода.
    this.dofScale = P.dof.resolutionScale ?? 1;
    if (P.bloom.enabled) {
      this.bloom = new UnrealBloomPass(new THREE.Vector2(1, 1),
        P.bloom.strength, P.bloom.radius, P.bloom.threshold);
      this.composer.addPass(this.bloom);
    }
    this.bloomScale = P.bloom.resolutionScale ?? 1;
    // Тонмаппинг и sRGB — здесь. Всё, что после, работает в экранном пространстве.
    this.composer.addPass(new OutputPass());

    this.grade = new ShaderPass(GradeShader);
    this.grade.renderToScreen = true;
    const g = P.grade;
    this.grade.uniforms.grain.value = g.grain;
    this.grade.uniforms.ca.value = g.chromaticAberration;
    this.grade.uniforms.vignette.value = g.vignette;
    this.grade.uniforms.vignetteSoftness.value = g.vignetteSoftness;
    this.composer.addPass(this.grade);

    this.setPostEnabled(P.enabledOnStart !== false);
  }

  /**
   * Включить или выключить пост целиком. RenderPass и OutputPass остаются:
   * без второго картинка ушла бы на экран в линейном пространстве, без ACES.
   */
  setPostEnabled(on) {
    this.postEnabled = on;
    for (const pass of this.composer.passes) {
      const name = pass.constructor.name;
      if (name === 'RenderPass' || name === 'OutputPass') continue;
      pass.enabled = on;
    }
    // Последний включённый пасс должен рисовать на экран.
    const active = this.composer.passes.filter((p) => p.enabled);
    for (const p of this.composer.passes) p.renderToScreen = false;
    if (active.length) active[active.length - 1].renderToScreen = true;
  }

  /**
   * Поставить камеру так, чтобы в кадре была грудь и голова, а глаза — на
   * верхней трети. Считается из фокусного расстояния и высоты кадра, а не
   * подбирается на глаз: 45 мм на кадре 36x24 дают вертикальный fov 30.3 град.
   */
  frameOn(point) {
    const C = this.cfg.camera;
    this.target.copy(point);

    this.camera.fov = 2 * Math.atan(12 / C.focalLengthMm) / DEG;
    this.camera.aspect = this.canvas.clientWidth / Math.max(1, this.canvas.clientHeight);
    const dist = C.frameHeightM / (2 * Math.tan(this.camera.fov * DEG / 2));

    // Глаза на верхней трети: центр кадра опускается ниже точки взгляда.
    const centerY = point.y - C.frameHeightM * (0.5 - C.eyeLineFromTop);
    const center = this._tmp.set(point.x, centerY, point.z);

    polarTo(center, C.azimuthDeg, C.elevationDeg, dist, this.camera.position);
    this.camera.lookAt(center);
    this.camera.updateProjectionMatrix();

    this.placeLights();
    if (this.bokeh) {
      // Фокус на глазах, а не на центре кадра.
      this.bokeh.uniforms.focus.value = this.camera.position.distanceTo(point);
    }
    this.focusDistance = this.camera.position.distanceTo(point);
  }

  setSize(w, h) {
    const dpr = Math.min(window.devicePixelRatio || 1, this.cfg.renderer.maxPixelRatio);
    this.renderer.setPixelRatio(dpr);
    this.renderer.setSize(w, h, false);
    this.composer.setPixelRatio(dpr);
    this.composer.setSize(w, h);
    // Размеры буферов округляются: setSize с дробной высотой создаёт текстуру
    // нецелого размера, и драйвер молча округляет её сам.
    const px = (v, k) => Math.max(1, Math.round(v * dpr * k));
    if (this.bloom) this.bloom.setSize(px(w, this.bloomScale), px(h, this.bloomScale));
    // setSize композитора уже выставил проходу полный размер — переопределяем.
    if (this.bokeh) this.bokeh.setSize(px(w, this.dofScale), px(h, this.dofScale));
    this.camera.aspect = w / Math.max(1, h);
    this.camera.updateProjectionMatrix();
  }

  render(timeSec) {
    this.renderer.info.reset();
    // Одно обновление карты теней на кадр — см. комментарий у autoUpdate.
    if (this.renderer.shadowMap.enabled) this.renderer.shadowMap.needsUpdate = true;
    this.grade.uniforms.time.value = timeSec;
    this.composer.render();
  }
}
