import os

def batch_replace_path(root_dir, old_str, new_str, file_extensions=None):
    """
    递归替换文件夹内文件中的字符串
    :param root_dir: 目标文件夹路径
    :param old_str: 待替换的旧路径
    :param new_str: 替换后的新路径
    :param file_extensions: 指定处理的文件后缀，例如 ['.yaml', '.py', '.json', '.sh']
    """
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            # 如果指定了后缀，则只处理匹配的文件
            if file_extensions and not any(file.endswith(ext) for ext in file_extensions):
                continue
            
            file_path = os.path.join(root, file)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if old_str in content:
                    new_content = content.replace(old_str, new_str)
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    print(f"✅ 已修改: {file_path}")
            except (UnicodeDecodeError, PermissionError):
                # 自动跳过二进制文件（如 .pth, .wav）或无权限文件
                continue

if __name__ == "__main__":
    # 填入你的项目根目录
    target_folder = "/data5/tyx/DiffSVS/data" 
    old_path = "/data7"
    new_path = "/data5"
    
    # 建议只针对配置文件和脚本进行操作，避免误伤数据文件
    extensions = ['.yaml', '.py', '.json', '.sh', '.txt', '.tsv']
    
    print(f"正在搜索并替换 {old_path} ...")
    batch_replace_path(target_folder, old_path, new_path, extensions)
    print("完成！")
