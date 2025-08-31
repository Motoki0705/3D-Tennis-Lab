import torch

from development.ball_tracking.ball_heatmap.infer import argmax_with_offset, upscale_coords


def test_argmax_with_offset_and_upscale():
    b, Hs, Ws, stride = 2, 8, 8, 4
    hmap = torch.zeros((b, 1, Hs, Ws), dtype=torch.float32)
    offs = torch.zeros((b, 2, Hs, Ws), dtype=torch.float32)
    # Peak at (5,6) with dx,dy
    y, x = 5, 6
    hmap[:, :, y, x] = 10.0
    offs[:, 0, y, x] = 0.4
    offs[:, 1, y, x] = 0.2

    coords_hm, scores = argmax_with_offset(hmap, offs)
    assert coords_hm.shape == (b, 2)
    assert torch.allclose(coords_hm[0], torch.tensor([x + 0.4, y + 0.2]))
    assert scores.min() == 10.0

    coords_img = upscale_coords(coords_hm, stride)
    assert torch.allclose(coords_img[0], torch.tensor([(x + 0.4) * stride, (y + 0.2) * stride]))
