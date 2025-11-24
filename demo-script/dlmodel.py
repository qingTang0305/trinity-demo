from huggingface_hub import snapshot_download
# 指定目标文件夹（绝对路径）
local_dir2 = "/home/ecs-user/models/Qwen2.5-0.5B-Instruct"

snapshot_download(
    repo_id="Qwen/Qwen2.5-0.5B-Instruct",  # 模型ID
    local_dir=local_dir2,               # 保存到此目录
    local_dir_use_symlinks=False,      # 避免软链接，直接复制文件（推荐）
    resume_download=True,              # 断点续传
)

# 指定目标文件夹（绝对路径）
local_dir = "/home/ecs-user/models/Qwen2-7B-Instruct"

snapshot_download(
    repo_id="Qwen/Qwen2-7B-Instruct",  # 模型ID
    local_dir=local_dir,               # 保存到此目录
    local_dir_use_symlinks=False,      # 避免软链接，直接复制文件（推荐）
    resume_download=True,              # 断点续传
)

print("finished")