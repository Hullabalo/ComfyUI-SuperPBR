import math
import torch
import torch.nn.functional as F


def _tile_grid(size, tile, stride):
    """Number of tiles needed to cover `size`, and the padded extent."""
    count = max(1, math.ceil((size - tile) / stride) + 1)
    return count, (count - 1) * stride + tile


def _split_batch(x_bchw, tile, stride, ny, nx):
    """[B,C,H,W] -> [nt*B,C,tile,tile], tile index major (row-major), batch minor."""
    tiles = [x_bchw[:, :, i * stride:i * stride + tile, j * stride:j * stride + tile]
             for i in range(ny) for j in range(nx)]
    return torch.cat(tiles, dim=0)


class SuperPBRTileSplit:
    """TileSplit cuts an image (optionally its mask) into an overlapping tile batch in row-major order.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "tile_size": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 64, "tooltip": "Tile size sent to SuperMat; 512 is what it was trained for."}),
                "overlap": ("INT", {"default": 128, "min": 0, "max": 1024, "tooltip": "Overlap between neighbouring tiles in px; must be < tile_size. Blend it away in TileStitch."}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Mask; split on the exact same grid."}),
            },
        }

    OUTPUT_TOOLTIPS = ["Tile batch (row-major).", "Mask split on the same grid (None if not connected).",
                       "Full mask-composited image padded to the tile canvas; wire into the decomposition node's anchor_image for the global-anchor blend.",
                       "Grid signature; wire into TileStitch so it uses the same geometry."]
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "TILESIG")
    RETURN_NAMES = ("image", "mask", "anchor_image", "tilesig")
    FUNCTION = "split"
    CATEGORY = "SuperPBR"

    def split(self, image, tile_size, overlap, mask=None):
        if overlap >= tile_size:
            raise ValueError("overlap must be smaller than tile_size")
        stride = tile_size - overlap
        B, H, W, _ = image.shape
        ny, ph = _tile_grid(H, tile_size, stride)
        nx, pw = _tile_grid(W, tile_size, stride)

        # Align the mask with the image grid. LoadImage emits a 64x64
        # placeholder MASK (2D [H,W], no batch dim) when the file has no alpha
        # channel; normalize to [B,H,W] first, then to the image grid.
        def align(x_bhw):
            if x_bhw.dim() == 2:
                x_bhw = x_bhw.unsqueeze(0)
            if x_bhw.shape[-2:] != (H, W):
                x_bhw = F.interpolate(x_bhw.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False).squeeze(1)
            return x_bhw

        def pad_and_split(x_bchw):
            x_bchw = F.pad(x_bchw, (0, pw - W, 0, ph - H), mode="replicate")
            return _split_batch(x_bchw, tile_size, stride, ny, nx)

        image_tiles = pad_and_split(image.movedim(-1, 1)).movedim(1, -1)
        amask = align(mask) if mask is not None else None
        mask_tiles = pad_and_split(amask.unsqueeze(1)).squeeze(1) if amask is not None else None
        padded = F.pad(image.movedim(-1, 1), (0, pw - W, 0, ph - H), mode="replicate")
        if amask is not None:
            pm = F.pad(amask.unsqueeze(1), (0, pw - W, 0, ph - H), mode="replicate")
            padded = padded * (1.0 - pm) + 0.5 * pm
        sig = {"height": H, "width": W, "batch": B, "tile": tile_size, "overlap": overlap, "stride": stride, "ny": ny, "nx": nx}
        return (image_tiles, mask_tiles, padded.movedim(1, -1), sig)


def _feather_1d(n, i, count, overlap):
    """1 for the tile's exclusive interior, 0->1 linear ramp across each overlapped edge (edges flush with the image stay 1)."""
    w = torch.ones(n)
    if overlap > 0:
        ramp = torch.linspace(0.0, 1.0, overlap + 1, dtype=torch.float32)[1:]
        if i > 0:
            w[:overlap] = torch.minimum(w[:overlap], ramp)
        if i < count - 1:
            w[n - overlap:] = torch.minimum(w[n - overlap:], ramp.flip(0))
    return w


class SuperPBRTileStitch:
    """TileStitch reassembles the decomposed maps with linear feather blending over the overlap margins.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "albedo": ("IMAGE", {"tooltip": "Albedo tile batch to reassemble."}),
                "orm": ("IMAGE", {"tooltip": "ORM tile batch on the same grid."}),
                "tilesig": ("TILESIG", {"tooltip": "Grid signature from TileSplit."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("albedo", "orm")
    FUNCTION = "stitch"
    CATEGORY = "SuperPBR"

    def stitch(self, albedo, orm, tilesig):
        t, o = tilesig["tile"], tilesig["overlap"]
        ny, nx, stride = tilesig["ny"], tilesig["nx"], tilesig["stride"]
        nt, B0, H, W = ny * nx, tilesig["batch"], tilesig["height"], tilesig["width"]

        def blend(tiles):
            x = tiles.movedim(-1, 1).to(torch.float32)  # [nt*B0,C,t,t]
            if x.shape[0] != nt * B0 or x.shape[-1] != t:
                raise ValueError("tile batch does not match the tilesig grid ({} tiles of {}px expected, got {} of {}px)".format(nt * B0, t, x.shape[0], x.shape[-1]))
            x = x.view(nt, B0, *x.shape[1:]).permute(1, 2, 0, 3, 4).reshape(B0, -1, ny, nx, t, t)  # [B0,C,ny,nx,t,t]
            ph, pw = (ny - 1) * stride + t, (nx - 1) * stride + t
            canvas = torch.zeros(B0, x.shape[1], ph, pw, device=x.device, dtype=torch.float32)
            wsum = torch.zeros(1, 1, ph, pw, device=x.device, dtype=torch.float32)
            for i in range(ny):
                wy = _feather_1d(t, i, ny, o)
                for j in range(nx):
                    w = wy[:, None] * _feather_1d(t, j, nx, o)[None, :]  # [t,t]
                    window = (slice(i * stride, i * stride + t), slice(j * stride, j * stride + t))
                    canvas[(...,) + window] += x[:, :, i, j] * w
                    wsum[(...,) + window] += w
            out = canvas / wsum.clamp_min(1e-8)
            return out[..., :H, :W].to(tiles.dtype).movedim(1, -1)

        return (blend(albedo), blend(orm))
