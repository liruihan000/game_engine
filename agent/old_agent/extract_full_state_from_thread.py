
# extract_full_state_from_thread.py
from langsmith import Client
import os
import json
from dotenv import load_dotenv

load_dotenv()

client = Client()

# 🎯 指定一个成功运行的 Run ID
run_id = "0199f60c-8aff-7773-bebc-f24d26a9680d"  # 替换为您的

try:
    run = client.read_run(run_id)
    print(f"✅ 找到 Run: {run.name}")
    
    # 获取初始输入（完整的 State）
    initial_state = run.inputs
    
    print(f"\n📥 初始 State 字段:")
    for key in initial_state.keys():
        print(f"  - {key}")
    
    # 保存到文件供检查
    with open("extracted_state.json", "w") as f:
        json.dump(initial_state, f, indent=2, default=str)
    
    print(f"\n💾 已保存到 extracted_state.json")
    
    # 检查是否有缺失字段
    required_fields = [
        "copilotkit", "items", "current_phase_id", "player_states",
        "gameName", "dsl", "need_feed_back_dict", 
        "referee_conclusions", "roomSession",  "phase_history"
    ]
    
    missing = [f for f in required_fields if f not in initial_state]
    
    if missing:
        print(f"\n⚠️ 缺失字段: {missing}")
        print("\n💡 需要补充默认值")
    else:
        print(f"\n✅ 所有必需字段都存在")
        
        # 创建 Dataset
        dataset_name = "werewolf-from-real-thread"
        try:
            dataset = client.create_dataset(dataset_name=dataset_name)
        except:
            datasets = list(client.list_datasets(dataset_name=dataset_name))
            dataset = datasets[0]
        
        # 添加到 Dataset
        client.create_example(
            dataset_id=dataset.id,
            inputs=initial_state,
            metadata={"source_run_id": run_id}
        )
        
        print(f"\n✅ 已添加到 Dataset: {dataset_name}")
    
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()