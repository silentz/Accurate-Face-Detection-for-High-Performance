import torch
import model.loss

loss = model.loss.AInnoFaceLoss()

gt = torch.Tensor([
    [
        [1., 1., 1., 1.],
        [1., 1., 2., 2.],
    ],
    [
        [1., 1., 1., 1.],
        [1., 1., 2., 2.],
    ],
])

anchors = torch.Tensor([
    [
        [1., 1., 2., 2.],
        [2., 2., 3., 3.],
        [3., 3., 4., 4.],
    ],
    [
        [1., 1., 2., 2.],
        [2., 2., 3., 3.],
        [3., 3., 4., 4.],
    ],
])

r = loss(None, None, anchors, gt)
print(r)
