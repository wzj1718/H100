import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoImageProcessor  
from transformers import AutoModelForCausalLM
from transformers import AutoModelForVision2Seq,AutoProcessor
import shutil
import os
import shutil
import torch
from transformers import AutoProcessor, AutoTokenizer

def _blend_module_params_(target_module, source_module, alpha_vl=0.9,alpha_base=0.1):
    """
    - 自动处理 weight 和 bias（若存在）
    - 确保 dtype/device 对齐
    """

    # 用 state_dict 逐个参数融合，更稳健也能覆盖所有权重/偏置
    tgt_sd = target_module.state_dict()
    src_sd = source_module.state_dict()

    # 形状检查
    for k in tgt_sd.keys():
        if k not in src_sd:
            raise ValueError(f"源模块缺少参数 {k}")
        if tgt_sd[k].shape != src_sd[k].shape:
            raise ValueError(f"参数形状不匹配: {k}, tgt={tgt_sd[k].shape}, src={src_sd[k].shape}")

    with torch.no_grad():
        for k in tgt_sd.keys():
            t = tgt_sd[k]
            s = src_sd[k].to(dtype=t.dtype, device=t.device)
            tgt_sd[k].copy_(alpha_vl * t + alpha_base * s)

    # 回写到模块
    target_module.load_state_dict(tgt_sd, strict=True)


def replace_self_attn_from_base_model(
    vl_model,
    base_model,
    start_layer=24,
    end_layer=35,
    save_dir="./merged_qwen_vl3b",
    orig_vl_model_path="/root/autodl-tmp/models/Qwen3-VL-4B-Instruct",
    alpha_vl=0.9,alpha_base=0.1
):
    """
    将 VL 模型指定层的 self_attn 模块与 Base 模型对应层做权重融合：
        new_attn = alpha_vl * vl_attn + (1 - alpha_vl) * base_attn
    并将完整模型（含 tokenizer、chat_template）保存到 save_dir。

    Args:
        vl_model: Qwen2.5-VL-3B-Instruct 模型
        base_model: Qwen2.5-3B 模型
        start_layer: 起始层索引（包含）
        end_layer: 结束层索引（包含）
        save_dir: 保存路径
        orig_vl_model_path: 原始VL模型路径，用于复制processor/tokenizer文件
        alpha_vl: 融合时 VL 权重系数（默认 0.9），Base 权重系数为 1 - alpha_vl
    """
    vl_layers = vl_model.model.language_model.layers
    base_layers = base_model.model.layers

    assert len(vl_layers) == len(base_layers), \
        f"❌ 层数不匹配：VL有{len(vl_layers)}层，Base有{len(base_layers)}层"


    print(f"🔧 开始融合层 {start_layer}~{end_layer} 的 self_attn 模块...")
    print(f"📊 总层数: {len(vl_layers)}")
    blended_layers = []

    with torch.no_grad():
        for i in range(start_layer, end_layer + 1):
            vl_attn = vl_layers[i].self_attn
            base_attn = base_layers[i].self_attn

            # 可选：最关键的 q/k/v/o 投影检查（提前发现结构差异）
            for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                assert hasattr(vl_attn, name) and hasattr(base_attn, name), f"{name} 不存在于 self_attn 中"
                v_mod = getattr(vl_attn, name)
                b_mod = getattr(base_attn, name)
                assert v_mod.weight.shape == b_mod.weight.shape, f"{name}.weight 形状不匹配"
                if hasattr(v_mod, "bias") and v_mod.bias is not None:
                    assert (b_mod.bias is not None) and (v_mod.bias.shape == b_mod.bias.shape), f"{name}.bias 形状不匹配"

            # 融合整个 self_attn 的 state_dict（覆盖所有子参数，包含 bias）
            _blend_module_params_(vl_attn, base_attn, alpha_vl=alpha_vl,alpha_base=alpha_base)
            blended_layers.append(i)

    print(f"🎯 成功融合 {len(blended_layers)} 层：{blended_layers}")

    # === 保存模型 ===
    os.makedirs(save_dir, exist_ok=True)
    print(f"💾 正在保存模型权重到：{save_dir}")
    vl_model.save_pretrained(save_dir)
    print("✅ 模型权重保存完成！")

    # === 同步保存 tokenizer / processor / chat_template ===
    print("📦 正在复制 tokenizer / processor / chat_template.json ...")
    processor = AutoProcessor.from_pretrained(orig_vl_model_path)
    tokenizer = AutoTokenizer.from_pretrained(orig_vl_model_path)

    processor.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    # 手动复制 chat_template.json（有时不会被 processor.save_pretrained 自动包含）
    src_template = os.path.join(orig_vl_model_path, "chat_template.json")
    dst_template = os.path.join(save_dir, "chat_template.json")
    if os.path.exists(src_template):
        shutil.copy(src_template, dst_template)
        print(f"✅ 已复制 chat_template.json 到 {dst_template}")
    else:
        print("⚠️ 未找到 chat_template.json，请确认原始模型目录中存在。")

    print(f"🎉 模型融合与保存全部完成：{save_dir}")
    return vl_model
import torch
from transformers import AutoModelForVision2Seq, AutoModelForCausalLM

qwen_vl_path = "/dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/di93pux/multimodal/models/Qwen3-VL-4B-Instruct"
qwen_base_path = "/dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/di93pux/multimodal/models/Qwen3-4B"

# === Base 模型只加载一次 ===
print("🚀 加载 Base 模型...")
base_model = AutoModelForCausalLM.from_pretrained(
    qwen_base_path,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="cpu"
)

# === 遍历 start_layer ===
for start_layer in range(19, 29):

    print(f"\n🔁 重新加载 VL 模型（start_layer={start_layer}）")
    vl_model = AutoModelForVision2Seq.from_pretrained(
        qwen_vl_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="cpu"
    )

    save_path = f"/dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/di93pux/multimodal/merged_models/merge_4/28/merge_{start_layer}--35+0.2base+0.8vl"

    replace_self_attn_from_base_model(
        vl_model,
        base_model,
        start_layer=start_layer,
        end_layer=35,
        save_dir=save_path,
        orig_vl_model_path=qwen_vl_path,
        alpha_vl=0.8,
        alpha_base=0.2
    )

    # === 关键：释放内存，防止累计占用 ===
    del vl_model
    torch.cuda.empty_cache()

