"""Mantis Gallery: a depth map from Depth Anything V2, run by the Studio's own Python (not ComfyUI's).

From your ComfyUI folder it only READS two things, as plain files: the checkpoint that
comfyui_controlnet_aux keeps (ckpts/.../depth_anything_v2_vit?.pth) and that node's model code (so the
weights load into exactly the network they were trained for). Nothing in ComfyUI or its venv is run,
imported into ComfyUI, installed or written. It runs on the GPU in half precision (about 0.7 GB, for a
second or two, freed when it exits); if the GPU hasn't room next to a running ComfyUI it finishes on the CPU.

    python mg_depthgen.py --comfy <ComfyUI folder> --in photo.png --out depth.png [--device cuda|cpu]

Needs torch in this Python. Writes a 16-bit PNG, near = white.
"""
import argparse
import importlib.util
import sys
import types
from pathlib import Path

CFG = {"vits": ([48, 96, 192, 384], 64), "vitb": ([96, 192, 384, 768], 128),
       "vitl": ([256, 512, 1024, 1024], 256), "vitg": ([1536, 1536, 1536, 1536], 384)}


def find(comfy):
    comfy = Path(comfy)
    aux = next((d for d in (comfy / "custom_nodes").iterdir() if d.is_dir() and "controlnet_aux" in d.name.lower()), None)
    if aux is None:
        sys.exit("comfyui_controlnet_aux isn't in that ComfyUI's custom_nodes")
    roots = [aux / "ckpts"]
    cfg = aux / "config.yaml"
    if cfg.exists():                                   # a custom checkpoint folder, if it names one
        for line in cfg.read_text(encoding="utf-8", errors="replace").splitlines():
            k, _, v = line.partition(":")
            if k.strip() == "annotator_ckpts_path" and v.strip():
                p = Path(v.strip().strip("'\""))
                roots.insert(0, p if p.is_absolute() else aux / p)
    ck = None
    for r in roots:
        if r.is_dir():
            hits = sorted(r.rglob("depth_anything_v2_vit*.pth"), key=lambda p: "vitl" not in p.name)
            if hits:
                ck = hits[0]
                break
    if ck is None:
        sys.exit("no depth_anything_v2_vit*.pth under controlnet_aux's checkpoints (run its Depth Anything V2 node once in ComfyUI)")
    code = aux / "src" / "custom_controlnet_aux" / "depth_anything_v2"
    if not (code / "dpt.py").exists():
        sys.exit(f"couldn't find Depth Anything V2's model code at {code}")
    return ck, code


def load_model(code, ck, device):
    import torch
    for mod in ("cv2", "torchvision", "torchvision.transforms"):     # only its image-prep helpers use these
        try:
            importlib.import_module(mod)
        except ImportError:
            m = types.ModuleType(mod)
            m.Compose = lambda *a, **k: None
            sys.modules[mod] = m
    # its folder as a package, without running controlnet_aux's own package setup (only model files load)
    pk_root = "mg_dav2"
    for name, path in ((pk_root, code),):
        m = types.ModuleType(name); m.__path__ = [str(path)]; sys.modules[name] = m
    spec = importlib.util.spec_from_file_location(f"{pk_root}.dpt", code / "dpt.py")
    dpt = importlib.util.module_from_spec(spec)
    sys.modules[f"{pk_root}.dpt"] = dpt
    spec.loader.exec_module(dpt)
    enc = next((e for e in CFG if e in ck.name), "vitl")
    oc, feat = CFG[enc]
    model = dpt.DepthAnythingV2(encoder=enc, features=feat, out_channels=oc)
    sd = torch.load(str(ck), map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    return model.to(device).eval(), enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--size", type=int, default=518)
    a = ap.parse_args()
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image

    dev = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"
    ck, code = find(a.comfy)
    model, enc = load_model(code, ck, "cpu")
    im = Image.open(a.inp).convert("RGB")
    w, h = im.size
    k = a.size / min(w, h)                              # short side ≥ size, both a multiple of 14
    nw, nh = max(14, int(round(w * k / 14)) * 14), max(14, int(round(h * k / 14)) * 14)
    x = np.asarray(im.resize((nw, nh), Image.BICUBIC), np.float32) / 255.0
    x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    t = torch.from_numpy(x.transpose(2, 0, 1))[None]

    def infer(device):
        m = model.to(device)
        tt = t.to(device)
        with torch.no_grad():
            if device == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    d = m(tt)
            else:
                d = m(tt)
            return F.interpolate(d.float()[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0].cpu().numpy()
    try:
        d = infer(dev)
    except RuntimeError as e:                           # (out of GPU memory next to ComfyUI: finish on the CPU)
        if dev != "cuda" or "out of memory" not in str(e).lower():
            raise
        torch.cuda.empty_cache()
        dev = "cpu (the GPU was full)"
        d = infer("cpu")
    lo, hi = np.percentile(d, 0.5), np.percentile(d, 99.5)        # (inverse depth: near is large → white)
    d = np.clip((d - lo) / max(1e-9, hi - lo), 0, 1)
    Image.fromarray((d * 65535).astype(np.uint16)).save(a.out)
    print(f"ok depth anything v2 {enc} on {dev}: {a.out}")


if __name__ == "__main__":
    main()
