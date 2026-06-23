# 接着上面的代码...

import torch
import torch.nn.functional as F
from einops import rearrange
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from arch_util import Attention  # 确保可以导入
import copy

# 在你的可视化文件中 (例如 visualizer_final.py)

# 这个列表将在每次可视化时被清空和填充
thread_local_storage = {'captured_attentions': []}


def forward_hook_factory(original_forward):
    """一个工厂函数，创建新的 forward 方法"""

    def new_forward(self, x, k_v):
        # --- 完全复制原始的 forward 逻辑 ---
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        # --- 核心修改：保存结果 ---
        # 保存一个克隆，防止后续操作影响张量
        thread_local_storage['captured_attentions'].append((attn.detach().clone().cpu(), (h, w)))

        # --- 继续原始的计算逻辑，确保模型输出不变 ---
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

    return new_forward










def visualize_attention_final(model, lr_input, save_dir="attention_maps_final"):
    """
    通过临时替换forward方法来可视化注意力图。
    """
    global thread_local_storage
    thread_local_storage['captured_attentions'] = []

    model_vis = copy.deepcopy(model)
    model_vis.eval()

    # 存储原始的 forward 方法
    original_forwards = {}

    # 1. 临时替换所有 Attention 模块的 forward 方法
    for i, module in enumerate(model_vis.modules()):
        if isinstance(module, Attention):
            # 保存原始方法
            original_forwards[i] = module.forward
            # 替换为我们的新方法
            module.forward = forward_hook_factory(module.forward).__get__(module, Attention)

    print(f"--- 临时替换了 {len(original_forwards)} 个 Attention 模块的 forward 方法 ---")

    # 2. 运行前向传播
    device = next(model_vis.parameters()).device
    lr_input = lr_input.to(device)
    with torch.no_grad():
        _ = model_vis(lr_input, None, None)

    # 3. 恢复原始的 forward 方法
    for i, module in enumerate(model_vis.modules()):
        if i in original_forwards:
            module.forward = original_forwards[i]
    print("--- 已恢复所有原始的 forward 方法 ---")

    captured_attentions = thread_local_storage['captured_attentions']
    if not captured_attentions:
        print("错误：未能捕获任何注意力图。")
        return

    # --- 后续的可视化代码与之前完全相同 ---
    print(f"\n--- 成功捕获了 {len(captured_attentions)} 个注意力图。开始可视化... ---")
    os.makedirs(save_dir, exist_ok=True)

    lr_image_np = np.transpose(lr_input.squeeze(0).cpu().numpy(), (1, 2, 0))
    lr_image_np = (lr_image_np - lr_image_np.min()) / (lr_image_np.max() - lr_image_np.min() + 1e-8)
    if lr_image_np.shape[2] == 1:
        lr_image_gray = lr_image_np.squeeze()
        lr_image_rgb_for_blending = np.concatenate([lr_image_np] * 3, axis=2)
    else:
        lr_image_gray = lr_image_np
        lr_image_rgb_for_blending = lr_image_np
    h_in, w_in = lr_image_gray.shape[:2]

    for i, (attn_map, (feat_h, feat_w)) in enumerate(captured_attentions):
        avg_attn_map = attn_map[0].mean(dim=0).mean(dim=0)

        if avg_attn_map.numel() != feat_h * feat_w:
            print(
                f"!!! 第 {i + 1} 层发生尺寸不匹配错误：元素数 {avg_attn_map.numel()} vs 目标尺寸 {feat_h}x{feat_w}。跳过。")
            continue

        heatmap = avg_attn_map.reshape(feat_h, feat_w).numpy()
        # ... (省略了绘图代码，与之前版本相同)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        cmap = plt.get_cmap('jet')
        heatmap_color_rgb = cmap(heatmap)[..., :3]
        pil_img = Image.fromarray((heatmap_color_rgb * 255).astype(np.uint8))
        pil_img_resized = pil_img.resize((w_in, h_in), Image.Resampling.LANCZOS)
        heatmap_resized = np.array(pil_img_resized) / 255.0
        superimposed_img = (lr_image_rgb_for_blending * 0.5) + (heatmap_resized * 0.5)
        save_path = os.path.join(save_dir, f"attention_layer_{i + 1}.png")
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(lr_image_gray, cmap='gray')
        axes[0].set_title("Low-Resolution Input")
        axes[0].axis('off')
        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title(f"Attention Heatmap ({feat_h}x{feat_w})")
        axes[1].axis('off')
        axes[2].imshow(superimposed_img)
        axes[2].set_title("Heatmap Superimposed")
        axes[2].axis('off')
        plt.suptitle(f"Attention Map from Layer {i + 1}", fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    print(f"\n可视化完成。图像已保存至: {os.path.abspath(save_dir)}")