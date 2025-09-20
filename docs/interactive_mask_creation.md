## Interactive Mask Creation on a Remote (Headless) Machine

If you're SSH’d into a headless box and OpenCV is installed in headless mode, `cv2.imshow`/windows won’t work for the interactive EasyHEC mask UI. Below are several practical ways to get interactive mask creation working. Options are ordered by convenience when using VS Code/Cursor Remote with automatic port forwarding.

### Option 1 — Web UI (recommended)

Run a small web app that serves the image and captures your clicks in the browser. VS Code/Cursor auto‑forwards remote ports to your local browser.

- **Gradio (quickest to wire up)**

  - Install: `uv add gradio`
  - Minimal app sketch (concept):

    ```python
    import gradio as gr
    import numpy as np
    from PIL import Image

    # images: list[np.ndarray(H,W,3)]
    # clicks collected as [(x, y, label)] where label in {1, -1}
    images = [...]  # load your frames
    clicks_per_image = [[] for _ in images]

    def on_click(evt: gr.SelectData, img_idx: int, pos_label: int):
        x, y = evt.index[0], evt.index[1]
        clicks_per_image[img_idx].append((x, y, pos_label))
        return f"Image {img_idx}: {len(clicks_per_image[img_idx])} points"

    with gr.Blocks() as demo:
        img_idx = gr.Number(value=0, label="Image Index", precision=0)
        pos_label = gr.Radio([1, -1], value=1, label="Label (1=pos, -1=neg)")
        status = gr.Textbox(label="Status")
        gallery = gr.Image(type="numpy")

        def load(idx):
            idx = int(idx)
            return Image.fromarray(images[idx])

        img_idx.change(load, inputs=img_idx, outputs=gallery)
        gallery.select(on_click, [img_idx, pos_label], status)

    demo.launch(server_name="0.0.0.0", server_port=7860)
    ```

  - Open VS Code “Ports” panel and click the forwarded `7860` port to open in your local browser.
  - After collecting points, run your SAM2 predictor (same logic as `InteractiveSegmentation`) and save masks as `numpy` arrays for the calibration script.

- **Streamlit (alternative)**
  - Install: `uv add streamlit`
  - Run: `uv run streamlit run app.py --server.port 7860 --server.address 0.0.0.0`

Notes:

- Bind to `0.0.0.0` on the remote; VS Code will forward the port securely to `localhost` on your machine.
- Keep the app private by relying on VS Code’s port forwarding; avoid opening the port publicly.

### Option 2 — Jupyter in the browser

If you prefer notebooks:

- Install: `uv add jupyterlab`
- Run on remote: `uv run jupyter lab --no-browser --port 8888 --ip 0.0.0.0`
- Use VS Code to forward port `8888`, open the URL locally.
- Implement a simple widget/matplotlib canvas to capture clicks and store `(x, y, label)` points per image, then run SAM2 to produce masks.

### Option 3 — Use labeling tools (no code)

If the auto-mask fails, manually annotate a few frames with off‑the‑shelf tools and save binary masks:

- **Label Studio** (web): install on remote, forward its port, draw polygons, export masks; convert to `(H, W)` boolean arrays.
- **Labelme** (desktop): rsync frames down, annotate locally, convert to masks, copy back.
  This is often the fastest way to get high‑quality masks without building UI.

### Option 4 — Enable OpenCV windows via X11/VNC (heavier ops setup)

If you want to keep using the existing OpenCV UI in EasyHEC:

- Replace headless OpenCV with GUI‑enabled build:
  - `uv pip uninstall opencv-python-headless` then `uv add opencv-python`
- Use one of:
  - SSH X11 Forwarding: `ssh -Y user@host` (requires XQuartz/macOS or X server on Windows/Linux). Latency can be high; ensure `xauth` is installed on the server.
  - VNC/noVNC: install a lightweight desktop (e.g., XFCE), `tigervnc`/`x11vnc`, and forward the VNC port; or use `noVNC` (websocket/web page) and forward the HTTP port via VS Code.

### Option 5 — Headless capture + local review loop

If a full UI is overkill, do a simple two‑step loop:

1. Save frames to disk on the remote (`results/.../base_camera/*.png`).
2. Pull them locally, annotate with any image editor to create binary masks (white=robot, black=background), push masks back as `.npy` files.

### Wiring masks back into the calibration script

The calibration script looks for `masks.npy` next to your images when `--use-previous-captures` is set. Produce a `(N, H, W)` boolean/0‑1 array and save it as:

```python
np.save("results/so101/so101_follower/base_camera/masks.npy", masks.astype(np.uint8))
```

Then run calibration with `--use-previous-captures` to skip data collection/masking.

### Troubleshooting

- If using OpenCV GUIs, make sure you don’t have `opencv-python-headless` installed; switch to `opencv-python`.
- For web UIs, if the page doesn’t load, verify the port is forwarded in VS Code and your app binds to `0.0.0.0`.
- For Jupyter, ensure you open the forwarded URL (token included) in your local browser.

### Recommendation

Start with the Gradio web UI. It’s minimal code, works great with VS Code’s auto port forwarding, and avoids GPU/GUI stack issues on headless servers. Once you’ve collected good masks, save them as `masks.npy` and re‑run calibration with `--use-previous-captures`.
