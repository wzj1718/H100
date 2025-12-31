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

def _blend_linear_(target_linear, source_linear, alpha_vl=0.9, alpha_base=0.1):
    """
    只融合 Linear 的 weight（以及 bias 若存在）
    target <- alpha_vl * target + alpha_base * source
    """
    with torch.no_grad():
        # weight
        tw = target_linear.weight
        sw = source_linear.weight.to(dtype=tw.dtype, device=tw.device)
        if tw.shape != sw.shape:
            raise ValueError(f"weight shape mismatch: tgt={tw.shape}, src={sw.shape}")
        tw.copy_(alpha_vl * tw + alpha_base * sw)

        # bias (optional)
        tb = getattr(target_linear, "bias", None)
        sb = getattr(source_linear, "bias", None)
        if tb is not None:
            if sb is None:
                raise ValueError("target has bias but source has no bias")
            sb = sb.to(dtype=tb.dtype, device=tb.device)
            if tb.shape != sb.shape:
                raise ValueError(f"bias shape mismatch: tgt={tb.shape}, src={sb.shape}")
            tb.copy_(alpha_vl * tb + alpha_base * sb)


def replace_self_attn_proj_from_base_model(
    vl_model,
    base_model,
    start_layer=24,
    end_layer=35,
    save_dir="./merged_qwen3vl",
    orig_vl_model_path=None,   # 建议传 Qwen3-VL 的原始目录
    alpha_vl=0.9,
    alpha_base=0.1,
):
    """
    只融合 VL 模型指定层 self_attn 的 q/k/v/o 投影：
        W_new = alpha_vl * W_vl + alpha_base * W_base

    保持 VL 的 q_norm/k_norm 不变；MLP 完全不动。
    """
    # 1) 拿到语言层
    vl_layers = vl_model.model.language_model.layers
    base_layers = base_model.model.layers

    assert len(vl_layers) == len(base_layers), \
        f"❌ 层数不匹配：VL有{len(vl_layers)}层，Base有{len(base_layers)}层"

    # 可选：避免你手滑把 alpha 设错
    #if abs((alpha_vl + alpha_base) - 1.0) > 1e-6:
     #   raise ValueError(f"alpha_vl + alpha_base should be 1.0, got {alpha_vl + alpha_base}")

    print(f"🔧 开始融合层 {start_layer}~{end_layer} 的 self_attn 投影(q/k/v/o)...")
    print(f"📊 总层数: {len(vl_layers)}")
    blended_layers = []

    with torch.no_grad():
        for i in range(start_layer, end_layer + 1):
            vl_attn = vl_layers[i].self_attn
            base_attn = base_layers[i].self_attn

            # 2) 只融合 q/k/v/o
            for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                if not (hasattr(vl_attn, name) and hasattr(base_attn, name)):
                    raise AttributeError(f"{name} 不存在于 self_attn 中 (layer={i})")
                v_mod = getattr(vl_attn, name)
                b_mod = getattr(base_attn, name)
                _blend_linear_(v_mod, b_mod, alpha_vl=alpha_vl, alpha_base=alpha_base)

            # 3) 明确声明：q_norm / k_norm 不动（这里不需要写代码，什么都不做就是“不动”）
            blended_layers.append(i)

    print(f"🎯 成功融合 {len(blended_layers)} 层：{blended_layers}")

    # === 保存模型 ===
    os.makedirs(save_dir, exist_ok=True)
    print(f"💾 正在保存模型权重到：{save_dir}")
    vl_model.save_pretrained(save_dir)
    print("✅ 模型权重保存完成！")

    # === 同步保存 tokenizer / processor / chat_template ===
    if orig_vl_model_path is not None:
        print("📦 正在复制 tokenizer / processor / chat_template.json ...")
        processor = AutoProcessor.from_pretrained(orig_vl_model_path)
        tokenizer = AutoTokenizer.from_pretrained(orig_vl_model_path)
        processor.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

        src_template = os.path.join(orig_vl_model_path, "chat_template.json")
        dst_template = os.path.join(save_dir, "chat_template.json")
        if os.path.exists(src_template):
            shutil.copy(src_template, dst_template)
            print(f"✅ 已复制 chat_template.json 到 {dst_template}")
        else:
            print("⚠️ 未找到 chat_template.json（不影响保存权重，但可能影响某些 chat 模板推理）")
    else:
        print("⚠️ 未提供 orig_vl_model_path：已仅保存模型权重。若要直接推理，请另外保存 processor/tokenizer。")

    print(f"🎉 模型融合与保存全部完成：{save_dir}")
    return vl_model


qwen_vl_path = "/dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/di93pux/multimodal/models/Qwen3-VL-8B-Instruct"
qwen_base_path = "/dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/di93pux/multimodal/models/Qwen3-8B"

base_save_root = "/dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/di93pux/multimodal/merged_models/merge_8B_only_qkvo/all"

#start_layers = list(range(19, 30))  # 19 ~ 28
start_layers=[24]
end_layer = 35

alpha_vl = 0.9
alpha_base = 0.3

print("🚀 正在加载 base model（只加载一次）...")
base_model = AutoModelForCausalLM.from_pretrained(
    qwen_base_path,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="cpu",
)

for start_layer in start_layers:
    save_path = (
        f"{base_save_root}/"
        f"merge_{start_layer}--{end_layer}+{alpha_base}base+{alpha_vl}vl"
    )

    print(f"\n🔁 开始处理 start_layer={start_layer}, end_layer={end_layer}")
    print(f"💾 保存路径：{save_path}")

    # ⚠️ 每次都重新加载 VL 模型，保证实验独立
    vl_model = AutoModelForVision2Seq.from_pretrained(
        qwen_vl_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )

    replace_self_attn_proj_from_base_model(
        vl_model=vl_model,
        base_model=base_model,
        start_layer=start_layer,
        end_layer=end_layer,
        save_dir=save_path,
        orig_vl_model_path=qwen_vl_path,
        alpha_vl=alpha_vl,
        alpha_base=alpha_base,
    )

    # 显式释放（CPU 内存也不小）
    del vl_model
    torch.cuda.empty_cache()

print("\n🎉 所有 start_layer ∈ [19, 27] 的 merge 实验已完成！")




