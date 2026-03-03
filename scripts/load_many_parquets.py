import os
import tqdm
from pathlib import Path
from datasets import load_dataset, interleave_datasets

base_path = "data/tokenizer_training_dataset"

# 1. 找出所有包含 parquet 文件的子文件夹
subdirs = [str(d) for d in Path(base_path).iterdir() if d.is_dir()]
print(f"检测到 {len(subdirs)} 个子数据集目录。")

all_streaming_datasets = []

for folder in subdirs:
    # 检查文件夹内是否有 parquet 文件，避免空跑
    if not list(Path(folder).glob("**/*.parquet")):
        continue
        
    try:
        # 针对每个文件夹加载，这样文件夹内部的 schema 是一致的
        ds = load_dataset(
            "parquet", 
            data_files=f"{folder}/**/*.parquet", 
            split="train", 
            streaming=True
        )
        
        # 统一映射逻辑
        def unify(example):
            # 这里的优先级可以根据你的观察调整
            text = example.get("text") or example.get("content") or ""
            return {"text": str(text)}

        # 只保留 text 列，确保不同文件夹合并时字段对齐
        ds = ds.map(unify, remove_columns=list(next(iter(ds)).keys()))
        all_streaming_datasets.append(ds)
        print(f"✅ 成功加载目录: {Path(folder).name}")
        
    except Exception as e:
        print(f"❌ 跳过目录 {Path(folder).name}，原因: {e}")

# 2. 使用 interleave_datasets 将所有流式数据集混合
# 这样你依然可以像操作一个大 dataset 一样操作它们
if all_streaming_datasets:
    combined_dataset = interleave_datasets(all_streaming_datasets)
    
    print("\n开始测试混合后的吞吐量...")
    count = 0
    for _ in tqdm.tqdm(combined_dataset):
        count += 1
else:
    print("没有成功加载任何数据集。")