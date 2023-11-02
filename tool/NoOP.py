import torch
from torch import nn
import numpy as np
import onnxruntime as ort

def compare_and_print(std_tensor, safe_tensor, detail=True, dense_grid=None, to_dense=False, do_sort=False):

    std_shape    = ' x '.join(map(str, std_tensor.shape))
    safe_shape  = ' x '.join(map(str, safe_tensor.shape))

    if detail:
        print("================ Compare Information =================")
        # print(f" std     Tensor: {std_shape}, {std_tensor.dtype} : {std_tensor}")
        # print(f" safe    Tensor: {safe_shape}, {safe_tensor.dtype} : {safe_tensor}")

    if np.cumprod(std_tensor.shape)[-1] != np.cumprod(safe_tensor.shape)[-1]:
        raise RuntimeError(f"Invalid compare with mismatched shape, {cpp_shape} < ----- > {safe_shape}")

    std_tensor   = std_tensor.reshape(-1).astype(np.float32)
    safe_tensor = safe_tensor.reshape(-1).astype(np.float32)

    if do_sort:
        std_tensor   = np.sort(std_tensor)
        safe_tensor = np.sort(safe_tensor)

    diff        = np.abs(std_tensor - safe_tensor)
    absdiff_max = diff.max().item()
    print(f"\033[31m[absdiff]: max:{absdiff_max}, sum:{diff.sum().item():.6f}, std:{diff.std().item():.6f}, mean:{diff.mean().item():.6f}\033[0m")
    if not detail:
        return

    print(f"std:   absmax:{np.abs(std_tensor).max().item():.6f}, min:{std_tensor.min().item():.6f}, std:{std_tensor.std().item():.6f}, mean:{std_tensor.mean().item():.6f}")
    print(f"safe:  absmax:{np.abs(safe_tensor).max().item():.6f}, min:{safe_tensor.min().item():.6f}, std:{safe_tensor.std().item():.6f}, mean:{safe_tensor.mean().item():.6f}")
    
    absdiff_p75 = absdiff_max * 0.75
    absdiff_p50 = absdiff_max * 0.50
    absdiff_p25 = absdiff_max * 0.25
    numel       = std_tensor.shape[0]
    num_p75     = np.sum(diff > absdiff_p75)
    num_p50     = np.sum(diff > absdiff_p50)
    num_p25     = np.sum(diff > absdiff_p25)
    num_p00     = np.sum(diff > 0)
    num_eq00    = np.sum(diff == 0)
    print(f"[absdiff > m75% --- {absdiff_p75:.6f}]: {num_p75 / numel * 100:.3f} %, {num_p75}")
    print(f"[absdiff > m50% --- {absdiff_p50:.6f}]: {num_p50 / numel * 100:.3f} %, {num_p50}")
    print(f"[absdiff > m25% --- {absdiff_p25:.6f}]: {num_p25 / numel * 100:.3f} %, {num_p25}")
    print(f"[absdiff > 0]: {num_p00 / numel * 100:.3f} %, {num_p00}")
    print(f"[absdiff = 0]: {num_eq00 / numel * 100:.3f} %, {num_eq00}")

    cpp_norm   = np.linalg.norm(std_tensor)
    torch_norm = np.linalg.norm(safe_tensor)
    sim        = (np.matmul(std_tensor, safe_tensor) / (cpp_norm * torch_norm))
    print(f"[cosine]: {sim * 100:.3f} %")
    print("======================================================")
    
    # np.testing.assert_almost_equal(std_tensor, safe_tensor, decimal=3)

class NoOP(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(1)

    def forward(self, x):
        x = x.view(10000, 64, 1)
        x = self.ln(x) + x
        x = x.view(10000, 64)
        return x

def createNoOP():
    model = NoOP()
    model.cuda()
    model.eval()

    dummy_input = torch.zeros((10000,64), dtype=torch.float32, device='cuda:0')

    with torch.no_grad():
      torch.onnx.export(model,
          dummy_input,
          "model/NoOP.onnx",
          export_params=True,
          opset_version=13,
          do_constant_folding=True,
          keep_initializers_as_inputs=True,
          input_names = ['input'],
          output_names = ['output'],
          )

def verifyNoOP():
    model_path = "model/NoOP.onnx"
    session = ort.InferenceSession(model_path)

    input = np.random.rand(10000,64).astype(np.float32)
    onnx_out = session.run(None, { "input": input })
    onnx_out = onnx_out[0]
    compare_and_print(onnx_out, input)

    # model = NoOP()
    # model.cuda()
    # model.eval()

    # input = torch.Tensor(np.random.rand(10000,64).astype(np.float32)).cuda()
    # print(input)
    # output = model(input)
    # print(output)

if __name__ == '__main__':
    createNoOP()
    verifyNoOP()