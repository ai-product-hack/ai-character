#!/usr/bin/env python3
"""Статистика по каналам metallicRoughness-текстур из GLB. Stdlib + sips.

    python3 mr_channels.py ../../data/model.glb

В glTF metallicRoughnessTexture кодирует шероховатость в G, металличность в B
(R обычно свободен или несёт occlusion). `metallicFactor`/`roughnessFactor`
не переопределяют текстуру, а **умножаются** на неё, поэтому фактор 1.0 сам по
себе ничего не говорит: смотреть надо в канал. Скрипт печатает min/среднее/max
и гистограмму по каждому каналу для каждой MR-текстуры, с именем материала.

JPEG распаковывается через `sips` (есть в macOS), PNG разбирается zlib'ом —
внешних питоновских зависимостей нет.
"""
import json, pathlib, struct, subprocess, sys, tempfile, zlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from inspect_glb import read_glb_bin


def decode_png(blob):
    """Возвращает (w, h, channels, bytearray пикселей). Только 8 бит на канал."""
    assert blob[:8] == b"\x89PNG\r\n\x1a\n", "не PNG"
    off, idat, w = 8, b"", None
    while off < len(blob):
        ln, typ = struct.unpack_from(">I4s", blob, off)
        data = blob[off + 8: off + 8 + ln]
        if typ == b"IHDR":
            w, h, depth, color = struct.unpack_from(">IIBB", data, 0)
            assert depth == 8, f"{depth} бит на канал не поддерживается"
            nch = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color]
        elif typ == b"IDAT":
            idat += data
        elif typ == b"IEND":
            break
        off += 12 + ln
    raw = zlib.decompress(idat)
    stride = w * nch
    out = bytearray(stride * h)
    prev = bytearray(stride)
    pos = 0
    for y in range(h):
        ft = raw[pos]; pos += 1
        line = bytearray(raw[pos: pos + stride]); pos += stride
        # Разворот фильтров PNG (спека, раздел 9.2).
        if ft == 1:
            for i in range(nch, stride):
                line[i] = (line[i] + line[i - nch]) & 0xFF
        elif ft == 2:
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 0xFF
        elif ft == 3:
            for i in range(stride):
                a = line[i - nch] if i >= nch else 0
                line[i] = (line[i] + ((a + prev[i]) >> 1)) & 0xFF
        elif ft == 4:
            for i in range(stride):
                a = line[i - nch] if i >= nch else 0
                b = prev[i]
                c = prev[i - nch] if i >= nch else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[i] = (line[i] + pr) & 0xFF
        out[y * stride:(y + 1) * stride] = line
        prev = line
    return w, h, nch, out


def decode_any(blob, mime):
    """JPEG прогоняется через sips в PNG — иначе в стдлибе его не развернуть."""
    if blob[:8] == b"\x89PNG\r\n\x1a\n":
        return decode_png(blob)
    with tempfile.TemporaryDirectory() as d:
        src = pathlib.Path(d) / "in.jpg"
        dst = pathlib.Path(d) / "out.png"
        src.write_bytes(blob)
        r = subprocess.run(["sips", "-s", "format", "png", str(src), "--out", str(dst)],
                           capture_output=True)
        if r.returncode != 0:
            raise RuntimeError(f"sips не смог развернуть {mime}: {r.stderr.decode()[:200]}")
        return decode_png(dst.read_bytes())


def stats(px, nch, ch, step=7):
    """min/avg/max и доля пикселей ниже 8/255 — «канал фактически нулевой»."""
    vals = px[ch::nch * step] if step > 1 else px[ch::nch]
    n = len(vals)
    lo, hi, s = 255, 0, 0
    near0 = near255 = 0
    for v in vals:
        s += v
        if v < lo: lo = v
        if v > hi: hi = v
        if v < 8: near0 += 1
        if v > 247: near255 += 1
    return {"min": lo, "max": hi, "avg": round(s / n, 1),
            "pct_lt8": round(100 * near0 / n), "pct_gt247": round(100 * near255 / n), "n": n}


def main(path):
    g, binc, _ = read_glb_bin(path)
    bvs, imgs, texs = g["bufferViews"], g["images"], g["textures"]
    out = []
    for mat in g.get("materials", []):
        pbr = mat.get("pbrMetallicRoughness", {})
        if "metallicRoughnessTexture" not in pbr:
            continue
        src = texs[pbr["metallicRoughnessTexture"]["index"]]["source"]
        bv = bvs[imgs[src]["bufferView"]]
        blob = binc[bv.get("byteOffset", 0): bv.get("byteOffset", 0) + bv["byteLength"]]
        w, h, nch, px = decode_any(blob, imgs[src].get("mimeType", ""))
        mf = pbr.get("metallicFactor", 1.0)
        rf = pbr.get("roughnessFactor", 1.0)
        rec = {"material": mat.get("name", "?"), "image": src, "w": w, "h": h,
               "channels": nch, "metallicFactor": mf, "roughnessFactor": rf,
               "R_occlusion": stats(px, nch, 0), "G_roughness": stats(px, nch, 1),
               "B_metallic": stats(px, nch, 2)}
        out.append(rec)
        print(f"\n=== {rec['material']}  (image #{src}, {w}x{h}, {nch} канала) ===")
        print(f"  metallicFactor={mf}  roughnessFactor={rf}  "
              f"(в glTF они УМНОЖАЮТСЯ на текстуру)")
        for ch, key in ((0, "R_occlusion"), (1, "G_roughness"), (2, "B_metallic")):
            s = rec[key]
            print(f"  {key:<12} min={s['min']:>3} avg={s['avg']:>6} max={s['max']:>3}  "
                  f"<8: {s['pct_lt8']:>3}%   >247: {s['pct_gt247']:>3}%")
        b = rec["B_metallic"]
        # Судим по среднему и доле пикселей, а не по max: у JPEG max задран
        # звоном кодека и одиночный выброс в 31 ничего не значит.
        zero = b["avg"] < 4 and b["pct_lt8"] >= 99
        rec["metallic_effectively_zero"] = zero
        print(f"  -> " + ("металличность фактически нулевая "
                          f"(avg={b['avg']}, {b['pct_lt8']}% пикселей <8; max={b['max']} — "
                          "звон JPEG) — metallicFactor=1.0 безвреден, это нормальный экспорт"
                          if zero else
                          f"металличность НЕ нулевая (avg={b['avg']}, max={b['max']}) — "
                          "фактор не трогать"))
    dst = pathlib.Path(__file__).resolve().parents[1] / "results" / "r5_mr_channels.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n-> {dst}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else sys.exit(__doc__))
