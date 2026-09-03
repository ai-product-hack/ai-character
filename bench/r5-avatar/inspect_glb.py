#!/usr/bin/env python3
"""Checks whether a GLB is usable as our talking head. Stdlib only.

    python3 inspect_glb.py avatar.glb [more.glb ...]

Pass criteria from the case: loads with a plain GLTFLoader, carries the 52 ARKit
morphs, small enough for the web. Reports exactly which ARKit names are missing,
because "has blendshapes" and "has the ones a viseme mapper needs" are different
claims.
"""
import json, pathlib, struct, sys

ARKIT52 = [
 "eyeBlinkLeft","eyeLookDownLeft","eyeLookInLeft","eyeLookOutLeft","eyeLookUpLeft",
 "eyeSquintLeft","eyeWideLeft","eyeBlinkRight","eyeLookDownRight","eyeLookInRight",
 "eyeLookOutRight","eyeLookUpRight","eyeSquintRight","eyeWideRight","jawForward",
 "jawLeft","jawRight","jawOpen","mouthClose","mouthFunnel","mouthPucker","mouthLeft",
 "mouthRight","mouthSmileLeft","mouthSmileRight","mouthFrownLeft","mouthFrownRight",
 "mouthDimpleLeft","mouthDimpleRight","mouthStretchLeft","mouthStretchRight",
 "mouthRollLower","mouthRollUpper","mouthShrugLower","mouthShrugUpper","mouthPressLeft",
 "mouthPressRight","mouthLowerDownLeft","mouthLowerDownRight","mouthUpperUpLeft",
 "mouthUpperUpRight","browDownLeft","browDownRight","browInnerUp","browOuterUpLeft",
 "browOuterUpRight","cheekPuff","cheekSquintLeft","cheekSquintRight","noseSneerLeft",
 "noseSneerRight","tongueOut",
]
# The eight visemes a TTS-alignment mapper actually drives (R4 level 2).
VISEME_CRITICAL = ["jawOpen","mouthClose","mouthFunnel","mouthPucker","mouthStretchLeft",
                   "mouthStretchRight","mouthRollLower","mouthRollUpper","mouthUpperUpLeft",
                   "mouthLowerDownLeft","tongueOut"]

# Полный набор Oculus/OVR LipSync. Если он есть в экспорте, матрица 13x52 не нужна:
# висемы кладутся один в один, а ARKit остаётся для эмоции и микроповедения.
OCULUS15 = ["viseme_sil","viseme_PP","viseme_FF","viseme_TH","viseme_DD","viseme_kk",
            "viseme_CH","viseme_SS","viseme_nn","viseme_RR","viseme_aa","viseme_E",
            "viseme_I","viseme_O","viseme_U"]
# Кости, без которых не сделать микроповедение (шея/голова) и взгляд (глаза).
BONES_WANTED = ["Neck","Head","LeftEye","RightEye","Spine2","LeftShoulder","RightShoulder"]


def read_glb(path):
    b = pathlib.Path(path).read_bytes()
    if b[:4] != b"glTF":
        return json.loads(b.decode()), len(b)          # plain .gltf
    ver, total = struct.unpack_from("<II", b, 4)
    off, js = 12, None
    while off < len(b):
        clen, ctype = struct.unpack_from("<II", b, off)
        chunk = b[off + 8: off + 8 + clen]
        if ctype == 0x4E4F534A:
            js = json.loads(chunk.decode("utf-8"))
        off += 8 + clen + ((4 - clen % 4) % 4 if clen % 4 else 0)
    return js, len(b)


def morph_names(g):
    """glTF stores morph names in mesh.extras.targetNames (the de-facto place
    exporters use); collect from every mesh and every primitive."""
    names, per_mesh = set(), {}
    for m in g.get("meshes", []):
        tn = (m.get("extras") or {}).get("targetNames") or []
        n_targets = max((len(p.get("targets", [])) for p in m.get("primitives", [])), default=0)
        per_mesh[m.get("name", "?")] = (len(tn), n_targets)
        names.update(tn)
    return names, per_mesh


def tri_count(g):
    n = 0
    for m in g.get("meshes", []):
        for p in m.get("primitives", []):
            if "indices" in p:
                n += g["accessors"][p["indices"]]["count"] // 3
    return n


def read_glb_bin(path):
    """То же, что read_glb, но отдаёт ещё и BIN-чанк — он нужен, чтобы прочитать
    заголовки картинок и узнать реальные размеры текстур без внешних библиотек."""
    b = pathlib.Path(path).read_bytes()
    if b[:4] != b"glTF":
        return json.loads(b.decode()), b"", len(b)
    off, js, binc = 12, None, b""
    while off < len(b):
        clen, ctype = struct.unpack_from("<II", b, off)
        chunk = b[off + 8: off + 8 + clen]
        if ctype == 0x4E4F534A:
            js = json.loads(chunk.decode("utf-8"))
        elif ctype == 0x004E4942:
            binc = chunk
        off += 8 + clen + ((4 - clen % 4) % 4 if clen % 4 else 0)
    return js, binc, len(b)


def image_size(blob):
    """Ширина/высота из заголовка PNG или JPEG. Стдлиб, без Pillow."""
    if blob[:8] == b"\x89PNG\r\n\x1a\n":
        w, h = struct.unpack_from(">II", blob, 16)
        return w, h, "png"
    if blob[:2] == b"\xff\xd8":
        i = 2
        while i < len(blob) - 9:
            if blob[i] != 0xFF:
                i += 1
                continue
            marker, seglen = blob[i + 1], struct.unpack_from(">H", blob, i + 2)[0]
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB):
                h, w = struct.unpack_from(">HH", blob, i + 5)
                return w, h, "jpeg"
            i += 2 + seglen
    return 0, 0, "?"


def textures(g, binc):
    """Список текстур: имя, формат, разрешение, вес. Тяжёлые текстуры — первое,
    что режут, если не держатся 60 FPS."""
    out = []
    for im in g.get("images", []):
        bv = g["bufferViews"][im["bufferView"]] if "bufferView" in im else None
        blob = binc[bv.get("byteOffset", 0): bv.get("byteOffset", 0) + bv["byteLength"]] if bv else b""
        w, h, fmt = image_size(blob)
        out.append({"name": im.get("name", "?"), "mime": im.get("mimeType", fmt),
                    "w": w, "h": h, "bytes": len(blob)})
    return out


def skeleton(g):
    """Есть ли скелет и где кости шеи, головы и глаз."""
    nodes = g.get("nodes", [])
    names = [n.get("name", "?") for n in nodes]
    found = {b: names.index(b) for b in BONES_WANTED if b in names}
    joints = g.get("skins", [{}])[0].get("joints", []) if g.get("skins") else []
    return {"nodes": len(nodes), "skins": len(g.get("skins", [])), "joints": len(joints),
            "bones_found": found,
            "bones_missing": [b for b in BONES_WANTED if b not in found]}


def report(path):
    g, binc, size = read_glb_bin(path)
    names, per_mesh = morph_names(g)
    per_mesh_names = {m.get("name", "?"): ((m.get("extras") or {}).get("targetNames") or [])
                      for m in g.get("meshes", [])}
    tex = textures(g, binc)
    skel = skeleton(g)
    low = {n.lower(): n for n in names}
    have = [a for a in ARKIT52 if a.lower() in low]
    missing = [a for a in ARKIT52 if a.lower() not in low]
    vis_missing = [v for v in VISEME_CRITICAL if v.lower() not in low]
    ext = g.get("extensionsUsed", [])
    print(f"\n=== {pathlib.Path(path).name} ===")
    print(f"размер: {size/1e6:.2f} МБ | треугольников: {tri_count(g):,} | "
          f"мешей: {len(g.get('meshes',[]))} | текстур: {len(g.get('images',[]))}")
    print(f"расширения: {', '.join(ext) or 'нет'}")
    vrm = [e for e in ext if e.startswith("VRM")]
    if vrm:
        print(f"  !! это VRM ({', '.join(vrm)}), а не обычный glTF — "
              f"GLTFLoader покажет меш, но выражения потребуют @pixiv/three-vrm")
    print(f"морфов всего: {len(names)}")
    for mesh, (tn, tg) in per_mesh.items():
        flag = "" if tn == tg else f"  <-- targetNames={tn} но targets={tg}, имена потеряны"
        print(f"  меш «{mesh}»: {tg} морф-таргетов{flag}")
    print(f"ARKit 52: {len(have)}/52")
    print(f"критичные для висем: {len(VISEME_CRITICAL)-len(vis_missing)}/{len(VISEME_CRITICAL)}"
          + (f" — НЕТ: {', '.join(vis_missing)}" if vis_missing else ""))
    if missing:
        print(f"нет из ARKit ({len(missing)}): {', '.join(missing[:12])}"
              + (" …" if len(missing) > 12 else ""))

    ovr_have = [v for v in OCULUS15 if v.lower() in low]
    ovr_missing = [v for v in OCULUS15 if v.lower() not in low]
    print(f"висемы Oculus: {len(ovr_have)}/15"
          + (f" — НЕТ: {', '.join(ovr_missing)}" if ovr_missing else " — маппинг 1:1, матрица не нужна"))
    extra = sorted(n for n in names if n not in ARKIT52 and n not in OCULUS15)
    if extra:
        print(f"сверх ARKit+Oculus ({len(extra)}): {', '.join(extra)}")

    print(f"скелет: {skel['nodes']} нод, {skel['joints']} костей в скине")
    print(f"  нужные кости: {', '.join(f'{k}=#{v}' for k, v in skel['bones_found'].items()) or 'нет'}")
    if skel["bones_missing"]:
        print(f"  НЕТ костей: {', '.join(skel['bones_missing'])}")

    tex_bytes = sum(t["bytes"] for t in tex)
    print(f"текстуры: {len(tex)} шт., {tex_bytes/1e6:.2f} МБ ({tex_bytes/size*100:.0f}% файла)")
    for t in sorted(tex, key=lambda t: -t["bytes"])[:8]:
        print(f"  {t['w']}x{t['h']} {t['bytes']/1e6:5.2f} МБ  {t['mime']:<10} {t['name']}")

    ok = len(have) >= 52 and size < 25e6 and not vrm
    print(f"ВЕРДИКТ: {'ГОДЕН' if ok else 'НЕ ГОДЕН'} "
          f"(нужно: 52 ARKit-морфа, обычный glTF, < 25 МБ)")
    return {"file": str(path), "size_mb": round(size/1e6, 2), "tris": tri_count(g),
            "morphs_total": len(names), "arkit_have": len(have),
            "arkit_missing": missing, "viseme_missing": vis_missing,
            "oculus_have": ovr_have, "oculus_missing": ovr_missing, "extra_morphs": extra,
            "meshes": {k: {"count": len(v), "names": v} for k, v in per_mesh_names.items()},
            "skeleton": skel, "textures": tex, "textures_mb": round(tex_bytes/1e6, 2),
            "extensions": ext, "is_vrm": bool(vrm), "verdict_ok": ok}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    out = [report(p) for p in sys.argv[1:]]
    dst = pathlib.Path(__file__).resolve().parents[1] / "results" / "r5_glb_inspect.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n-> {dst}")
