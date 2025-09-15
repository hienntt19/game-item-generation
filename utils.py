import os
from safetensors.torch import load_file

def fix_lora_keys(lora_path):
    lora_dict = load_file(lora_path)
    
    fixed_dict = {}
    for key, value in lora_dict.items():
        if any(key.startswith(prefix) for prefix in ['down_blocks.', 'mid_block.', 'up_blocks.']):
            fixed_key = f"unet.{key}"
        else:
            fixed_key = key
        fixed_dict[fixed_key] = value
    
    print(f"Fixed {len(fixed_dict)} keys")
    return fixed_dict