import math

import torch
import torch.nn.functional as F

import comfy.utils
import folder_paths

from .model import LATENT_SCALING_FACTOR, V_PREDICTION_SCALE, SuperMatModel


folder_paths.folder_names_and_paths["supermat_models"] = (folder_paths.get_folder_paths("diffusion_models"), (".pth", ".safetensors"))


class SuperPBRModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (folder_paths.get_filename_list("supermat_models"), {"tooltip": "Base SuperMat checkpoint."}),
            }
        }

    RETURN_TYPES = ("SUPERMAT",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "SuperPBR"

    def load(self, model):
        return (SuperMatModel(folder_paths.get_full_path("supermat_models", model)),)


def _gauss_blur(x_bchw, sigma):
    """Separable Gaussian blur with reflect padding (no dark border, no halos
    from zero-padding). sigma is in pixels."""
    if sigma <= 0:
        return x_bchw
    C = x_bchw.shape[1]
    r = int(3 * sigma + 0.5)
    xs = torch.arange(-r, r + 1, device=x_bchw.device, dtype=x_bchw.dtype)
    k1 = torch.exp(-xs * xs / (2 * sigma * sigma))
    k1 /= k1.sum()
    k = k1.view(1, 1, 1, -1).expand(C, 1, 1, -1)  # depthwise horizontal (along W)
    xp = F.pad(x_bchw, (r, r, 0, 0), mode="reflect")
    x = F.conv2d(xp, k, groups=C)
    k = k1.view(1, 1, -1, 1).expand(C, 1, -1, 1)  # depthwise vertical (along H)
    xp = F.pad(x, (0, 0, r, r), mode="reflect")
    return F.conv2d(xp, k, groups=C)

class SuperPBRDecompose:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("SUPERMAT",),
                "vae": ("VAE",),
                "resolution": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 64}),
                "batch_chunk": ("INT", {"default": 8, "min": 1, "max": 16, "tooltip": "Images decomposed per UNet/VAE pass; larger batches are processed in chunks of this size and concatenated back — same output, capped peak VRAM. Lower it if you OOM on big tile grids."}),
            },
            "optional": {
                "clip": ("CLIP", {"tooltip": "SD 2.1 text encoder, only used for the fixed empty-prompt context. Optional; a zero context (negligible output shift) is used when not connected."}),
                "mask": ("MASK", {"tooltip": "Follows LoadImage's MASK convention: 0 = subject (kept), 1 = background (composited to 0.5 gray)."}),
                "anchor_image": ("IMAGE", {"tooltip": "Anchor from Tile Split's anchor_image output (mask-composited, padded to the tile canvas). It is decomposed once and its low frequencies anchor every tile, killing tile-to-tile colour drift."}),
                "tilesig": ("TILESIG", {"tooltip": "Grid signature from Tile Split; needed to crop the anchor on the tile grid."}),
            },
        }

    OUTPUT_TOOLTIPS = ["SuperMat albedo map in [0,1].",
        "ORM map: R=occlusion (AO), G=roughness, B=metallic. Split into separate maps with the ORM Split node."]
    RETURN_TYPES = ("IMAGE",) * 2
    RETURN_NAMES = ("albedo", "orm")
    FUNCTION = "decompose"
    CATEGORY = "SuperPBR"

    def decompose(self, image, model, vae, resolution, batch_chunk=8, clip=None, mask=None, anchor_image=None, tilesig=None):
        anchor_blend = anchor_image is not None and tilesig is not None
        # The anchor is upscaled back to the tile canvas and cropped on the tile grid;
        # each crop is blended 1:1 with its tile, so the tile pass must run at
        # tile_size (not resolution) for the shapes to line up.
        if anchor_blend:
            resolution = tilesig["tile"]
        # common_upscale expects BCHW, so MASK (which has no channel dim) needs a fake one
        image = comfy.utils.common_upscale(image.movedim(-1, 1), resolution, resolution, upscale_method="bilinear", crop="none").movedim(1, -1)
        if mask is not None:
            mask = comfy.utils.common_upscale(mask.unsqueeze(1), resolution, resolution, upscale_method="bilinear", crop="none").squeeze(1).unsqueeze(-1)
            image = image * (1.0 - mask) + 0.5 * mask
        context = clip.encode("") if clip is not None else None

        if not anchor_blend:
            out = [[], []]
            for k in range(0, image.shape[0], batch_chunk):
                for hi, m in enumerate(model.decode_heads(vae, model.predict(vae, image[k:k + batch_chunk], context))):
                    out[hi].append(m)
            return (torch.cat(out[0]), torch.cat(out[1]))

        # Global-anchor blend: low frequencies (colour/tone) come from the
        # anchor pass — identical across tiles, killing tile-to-tile drift —
        # high frequencies from the tile. In pixel space, not latent: averaging
        # SD-VAE latents lands off the decoder manifold (gray blotches).
        # sigma = 8 px keeps a tile of colour/tone in the low band.
        t, stride = tilesig["tile"], tilesig["stride"]
        ny, nx = tilesig["ny"], tilesig["nx"]
        ph, pw = (ny - 1) * stride + t, (nx - 1) * stride + t
        g = anchor_image.movedim(-1, 1)
        gs = t / max(ph, pw)  # aspect-preserving: tiles are square, the canvas is not
        gh, gw = int(round(g.shape[2] * gs / 8)) * 8, int(round(g.shape[3] * gs / 8)) * 8
        g = comfy.utils.common_upscale(g, gw, gh, upscale_method="lanczos", crop="none").movedim(1, -1)  # (width, height)

        anchors = []  # per head: [nt*B,t,t,3] low-passed pixel crops from the global pass
        for m in model.decode_heads(vae, model.predict(vae, g, context)):
            m = comfy.utils.common_upscale(m.movedim(-1, 1), pw, ph, upscale_method="lanczos", crop="none")  # [B,3,ph,pw]
            # Low-pass on the whole canvas, before cropping: blurring per tile would
            # reflect-pad differently at each tile edge and reintroduce tile-to-tile drift.
            m = _gauss_blur(m, 8.0).movedim(1, -1)
            anchors.append(torch.cat([m[:, i * stride:i * stride + t, j * stride:j * stride + t]
                                      for i in range(ny) for j in range(nx)], dim=0))
        sc = V_PREDICTION_SCALE / LATENT_SCALING_FACTOR
        out = [[], []]  # per head: list of chunks
        for k in range(0, image.shape[0], batch_chunk):
            heads = model.predict(vae, image[k:k + batch_chunk], context)
            for hi, h in enumerate(heads):
                tb = vae.decode(h * sc).movedim(-1, 1)
                ab = anchors[hi][k:k + batch_chunk].movedim(-1, 1)
                out[hi].append((ab + tb - _gauss_blur(tb, 8.0)).clamp(0.0, 1.0).movedim(1, -1))
        return (torch.cat(out[0]), torch.cat(out[1]))


class SuperPBRDetilt:
    """Removes the spurious tilt/keystone plane from a depth map so it displaces flat."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Depth/height map. The background (zero-displacement) plane is estimated and subtracted."}),
                "method": (["quadratic", "bilinear", "none"], {"tooltip": "quadratic: second-order surface fitted on the border ring — also removes the smooth bowl/dome curvature depth models leave on flat inputs. bilinear: background interpolated between the 4 averaged corner patches. none: pass the map through unchanged."}),
            },
        }
    
    OUTPUT_TOOLTIPS = ["De-tilted height/depth map in [0,1]: the original minus the fitted background plane, renormalized. Returned unchanged when method is 'none'."]
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "detilt"
    CATEGORY = "SuperPBR"

    def detilt(self, image, method):
        if method == "none":
            return (image,)
        d = image.mean(dim=-1)  # [B,H,W], gray
        B, H, W = d.shape
        yy, xx = torch.meshgrid(torch.arange(H, device=d.device), torch.arange(W, device=d.device), indexing="ij")
        X, Y = xx.float() / W, yy.float() / H  # [H,W]
        if method == "bilinear":
            p = max(1, min(H, W) // 128)  # corner patch size (~8px at 1024), corner pixels alone are noisy
            out = []
            for b in range(B):
                c = [d[b, :p, :p].mean(), d[b, :p, -p:].mean(), d[b, -p:, :p].mean(), d[b, -p:, -p:].mean()]  # tl tr bl br
                u, v = X[0].unsqueeze(0), Y[:, 0].unsqueeze(-1)  # [1,W], [H,1]
                bg = torch.lerp(torch.lerp(c[0], c[1], u), torch.lerp(c[2], c[3], u), v)
                out.append(d[b] - bg)
        else: # quadratic - second-order surface on the border ring (captures dome/bowl curvature)
            r = max(2, min(H, W) // 16)
            ring = (xx < r) | (xx >= W - r) | (yy < r) | (yy >= H - r)
            stride = max(1, int(math.sqrt(ring.sum().item()) // 150))
            if stride > 1:
                sub = torch.zeros_like(ring)
                sub[::stride, ::stride] = True
                ring = ring & sub
            xs, ys = xx[ring].float() / W, yy[ring].float() / H
            xs2, ys2, xys = xs * xs, ys * ys, xs * ys
            out = []
            for b in range(B):
                db = d[b][ring]
                m = torch.ones_like(db, dtype=torch.bool)
                for _ in range(4):
                    A = torch.stack([torch.ones_like(xs[m]), xs[m], ys[m], xs2[m], ys2[m], xys[m]], dim=1)
                    p = torch.linalg.lstsq(A, db[m].unsqueeze(1)).solution
                    res = (A @ p).squeeze(-1) - db[m]
                    keep = res.abs() < 2.5 * res.std()
                    m2 = torch.zeros_like(m)
                    m2[torch.nonzero(m, as_tuple=True)[0][keep]] = True
                    if m2.sum() <= 100: break
                    m = m2
                out.append(d[b] - (p[0] + p[1] * X + p[2] * Y + p[3] * X * X + p[4] * Y * Y + p[5] * X * Y))
        h = torch.stack(out)  # [B,H,W]
        rng = h.amax(dim=(1, 2), keepdim=True) - h.amin(dim=(1, 2), keepdim=True)
        flat = rng < 1e-6  # constant input: nothing to remove, return it unchanged (gray, not black)
        lo = torch.quantile(h.flatten(1), 0.001, dim=1).view(B, 1, 1)
        hi = torch.quantile(h.flatten(1), 0.999, dim=1).view(B, 1, 1)
        h = ((h - lo) / (hi - lo + 1e-8)).clamp(0.0, 1.0)
        h = torch.where(flat, image.mean(dim=-1), h).unsqueeze(-1)
        return (h.expand(-1, -1, -1, 3).contiguous(),)


class SuperPBRORMSplit:
    """Splits an ORM map (R=occlusion, G=roughness, B=metallic) into three gray images."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "orm": ("IMAGE", {"tooltip": "ORM map from SuperMat Material Decomposition (or stitched via Tile Stitch): R=occlusion (AO), G=roughness, B=metallic."}),
            },
        }

    OUTPUT_TOOLTIPS = [
        "Ambient occlusion (R channel). Blender renders its own AO at render time — use this for baking/export or subtle base-color darkening, not as a direct Principled input.",
        "Roughness (G channel).",
        "Metallic (B channel).",
    ]
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("occlusion", "roughness", "metallic")
    FUNCTION = "split"
    CATEGORY = "SuperPBR"

    def split(self, orm):
        return (orm[..., 0:1].expand(-1, -1, -1, 3).contiguous(),
                orm[..., 1:2].expand(-1, -1, -1, 3).contiguous(),
                orm[..., 2:3].expand(-1, -1, -1, 3).contiguous())
