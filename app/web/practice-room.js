import * as THREE from 'three';

/** A quiet architectural backdrop for the practice screen. Uses simple geometry,
 * no image downloads or extra render passes; the avatar's lighting stays intact. */
export function createPracticeRoom(look) {
  look.backdrop.visible = false;
  look.scene.background = new THREE.Color('#e3e9df');
  const room = new THREE.Group();
  room.name = 'practice-studio';
  const surface = color => new THREE.MeshBasicMaterial({color});
  const wall = surface('#e3e9df'), inset = surface('#d3dfd0');
  const trim = surface('#c5d2bf'), wood = surface('#c9c2aa');
  const box = (w, h, d, x, y, z, material) => {
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(w,h,d), material);
    mesh.position.set(x,y,z); room.add(mesh); return mesh;
  };
  box(8,6,.1,0,1,-1.5,wall);
  // Recessed wall panels and a soft daylight opening give the scene depth.
  box(.75,2.8,.06,-.9,1.5,-1.4,inset);
  const arch = new THREE.Mesh(new THREE.CircleGeometry(.375,48,0,Math.PI), inset);
  arch.position.set(-.9,2.9,-1.36); room.add(arch);
  box(.035,3.4,.03,-1.35,1.55,-1.32,trim);
  box(.035,3.4,.03,-.34,1.55,-1.32,trim);
  box(1.2,2.4,.04,.85,2.1,-1.35,surface('#f0f3e8'));
  for (const x of [.25,.85,1.45]) box(.025,2.4,.04,x,2.1,-1.3,wood);
  box(1.2,.025,.04,.85,2.1,-1.3,wood);
  box(1.5,.05,.3,.8,.88,-1.1,wood);
  // One restrained plant, with a deterministic silhouette.
  const stem = surface('#7f9376'), leaf = surface('#8ca180');
  const pot = new THREE.Mesh(new THREE.CylinderGeometry(.1,.07,.18,24),surface('#d3c6af'));
  pot.position.set(.62,.97,-.94);room.add(pot);
  box(.012,.48,.012,.62,1.28,-.94,stem);
  for(let i=0;i<5;i++) {
    const side = i%2 ? 1 : -1;
    const mesh = new THREE.Mesh(new THREE.SphereGeometry(1,12,8),leaf);
    mesh.scale.set(.085,.034,.025);
    mesh.position.set(.62+side*.065,1.1+i*.075,-.94);
    mesh.rotation.z=side*.5;room.add(mesh);
  }
  look.scene.add(room);
  return room;
}
