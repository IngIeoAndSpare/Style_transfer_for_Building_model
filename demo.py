"""Browser demo

using 1. python demo.py
using 2. python demo.py --gpu 1 --port 7861 --engine-port 8189 
using 3. python demo.py --gpu 1 --port 7861 --host 0.0.0.0 
"""

import os
import re
import sys
import glob
import time
import argparse
import traceback
import gradio as gr

from facadestyle import config
from facadestyle.engine import ensure_engine
from facadestyle.masks import WallWindowMasker
from facadestyle.pipeline import Pipeline


ROOT = os.path.dirname(os.path.abspath(__file__))
CUSTOM = "Custom"
CUSTOM_PROMPT = ("Photorealistic building facade. Preserve the original facade layout, window positions, proportions, "
                 "and perspective with no deformation.")


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1", help="address the UI listens on; 0.0.0.0 makes it reachable from other machines")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--gpu", type=int, default=None, help="CUDA device index; default: the current CUDA_VISIBLE_DEVICES")
    ap.add_argument("--engine-url", default=None, help="use a running engine instead of starting one")
    ap.add_argument("--engine-port", type=int, default=None, help="port of the engine this script starts (default 8188)")
    ap.add_argument("--runs", default="runs", help="where results are written")
    ap.add_argument("--examples", default=os.path.join(ROOT, "examples"), help="folder of example buildings")
    ap.add_argument("--share", action="store_true", help="also publish a temporary public Gradio link")
    return ap.parse_args()


def find_examples(root, cache_dir):
    import cv2
    from facadestyle.obj_io import load_building
    from facadestyle.preview import render_thumbnail
    out = []
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        objs = glob.glob(os.path.join(d, "*.obj"))
        if not os.path.isdir(d) or len(objs) != 1:
            continue
        mtls = glob.glob(os.path.join(d, "*.mtl"))
        textures = sorted(p for p in glob.glob(os.path.join(d, "*")) if p.lower().endswith((".png", ".jpg", ".jpeg"))
                          and os.path.basename(p) != "thumbnail.png")
        name = os.path.basename(d)
        thumb = os.path.join(d, "thumbnail.png")
        if not os.path.exists(thumb):
            thumb = os.path.join(cache_dir, f"{name}.png")
            if not os.path.exists(thumb):
                try:
                    os.makedirs(cache_dir, exist_ok=True)
                    cv2.imwrite(thumb, render_thumbnail(load_building(objs[0], mtls[0] if mtls else None, textures)))
                except Exception as e:
                    print(f"example {name} skipped: {e}")
                    continue
        label = name.split("_", 1)[1] if name.split("_", 1)[0].isdigit() and "_" in name else name
        out.append((label, objs[0], mtls[0] if mtls else None, textures, thumb))
    return out


def progress_fraction(msg, last):
    m = re.search(r"\[(\d+)/(\d+)\]", msg)
    done = (int(m.group(1)) - 1) / int(m.group(2)) if m else 0.0
    if "reconstruction" in msg:
        return 0.95
    if m and "stylization" in msg:
        return max(last, 0.3 + 0.65 * done)
    if m:
        return max(last, 0.3 * done)
    return last


def serve(args, engine):
    masker = WallWindowMasker(config.SAM2_CHECKPOINT)
    pipe = Pipeline(engine, masker)
    examples = find_examples(args.examples, os.path.join(args.runs, "_thumbnails"))
    presets = {p["label"]: p for p in config.PRESETS}
    first = config.PRESETS[0]

    def run(model_files, texture_files, choice, custom_image, custom_prompt, progress=gr.Progress()):
        model_files = [f if isinstance(f, str) else f.name for f in (model_files or [])]
        texture_files = [f if isinstance(f, str) else f.name for f in (texture_files or [])]
        objs = [f for f in model_files if f.lower().endswith(".obj")]
        mtls = [f for f in model_files if f.lower().endswith(".mtl")]
        if len(objs) != 1:
            raise gr.Error("Upload exactly one .obj file.")
        if not texture_files:
            raise gr.Error("Upload the texture atlas image(s) of the model.")
        if choice == CUSTOM:
            if not custom_image:
                raise gr.Error("Upload a style reference image for the custom style.")
            if not (custom_prompt or "").strip():
                raise gr.Error("Enter a prompt for the custom style.")
            style = ("custom", custom_image, custom_prompt.strip())
        else:
            p = presets[choice]
            style = (p["key"], p["image"], p["prompt"])
        name = os.path.splitext(os.path.basename(objs[0]))[0]
        out_dir = os.path.join(args.runs, time.strftime("%Y%m%d_%H%M%S") + "_" + name)
        state = {"frac": 0.0}

        def report(msg):
            state["frac"] = progress_fraction(msg, state["frac"])
            if msg.startswith(style[0] + " ") or msg.startswith(style[0] + ":"):
                msg = choice + msg[len(style[0]):]
            progress(state["frac"], desc=msg)

        try:
            result = pipe.run(objs[0], mtls[0] if mtls else None, texture_files, out_dir, [style], progress=report)
        except Exception as e:
            traceback.print_exc()
            raise gr.Error(str(e))
        key = style[0]
        failed = [f["facade"] for f in result["facades"]
                  if f["status"] == "failed" or f.get("styles", {}).get(key, {}).get("status") == "failed"]
        if failed:
            gr.Warning("Kept the original texture for: " + ", ".join(failed))
        out = result["styles"][key]
        return result.get("original_glb"), out.get("stylized_glb"), out["archive"]

    def pick(choice):
        custom = choice == CUSTOM
        p = presets.get(choice, first)
        return (gr.update(visible=not custom), gr.update(value=p["image"]), gr.update(value=p["prompt"]),
                gr.update(visible=custom))

    with gr.Blocks(title="Facade style transfer of atlas textures") as ui:
        gr.Markdown("## Facade-based style transfer of atlas textures for building models\n"
                    "Upload a textured building model and its texture atlas, and choose a style. The facades are "
                    "extracted, stylized with the style reference while their structure is kept, and written back "
                    "into the atlas.")
        with gr.Row():
            model_in = gr.File(label="Building model (.obj)", file_count="multiple", file_types=[".obj", ".mtl"])
            tex_in = gr.File(label="Texture atlas (.png or .jpg)", file_count="multiple", file_types=["image"])
        style_in = gr.Radio([p["label"] for p in config.PRESETS] + [CUSTOM], value=first["label"], label="Style")
        with gr.Row(visible=True) as preset_box:
            preset_image = gr.Image(value=first["image"], label="Style reference", interactive=False, height=240)
            preset_prompt = gr.Textbox(value=first["prompt"], label="Prompt", lines=5, interactive=False)
        with gr.Row(visible=False) as custom_box:
            custom_image = gr.Image(label="Style reference image", type="filepath", image_mode=None, sources=["upload"],
                                    height=240)
            custom_prompt = gr.Textbox(value=CUSTOM_PROMPT, label="Prompt", lines=5)
        style_in.change(pick, style_in, [preset_box, preset_image, preset_prompt, custom_box])
        btn = gr.Button("Apply style", variant="primary")
        with gr.Row():
            original = gr.Model3D(label="Input model")
            stylized = gr.Model3D(label="Stylized model")
        archive = gr.File(label="Stylized model package (zip)")
        inputs = [model_in, tex_in, style_in, custom_image, custom_prompt]
        outputs = [original, stylized, archive]
        btn.click(run, inputs, outputs)

        if examples:
            gr.Markdown("### Example building models\nClick a building to load it and run it with the selected style.")
            gallery = gr.Gallery(value=[(e[4], e[0]) for e in examples], columns=min(len(examples), 5), height=260,
                                 object_fit="contain", allow_preview=False, show_label=False)

            def load_example(evt: gr.SelectData):
                _, obj, mtl, textures, _ = examples[evt.index]
                return [obj] + ([mtl] if mtl else []), textures

            gallery.select(load_example, None, [model_in, tex_in]).then(run, inputs, outputs)

    ui.queue().launch(server_name=args.host, server_port=args.port, share=args.share,
                      allowed_paths=[os.path.abspath(args.runs), os.path.abspath(args.examples), config.STYLES_DIR])
    return 0


def main():
    args = parse()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    missing = config.missing_weights()
    if missing:
        print("missing weights (see README):\n  " + "\n  ".join(missing))
        return 1
    engine, proc = ensure_engine(args.engine_url, args.engine_port or config.ENGINE_PORT)
    try:
        return serve(args, engine)
    finally:
        if proc is not None:
            proc.terminate()


if __name__ == "__main__":
    sys.exit(main())
