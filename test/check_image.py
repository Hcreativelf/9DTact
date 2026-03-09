import os
import cv2

def check_image(file_path):
    """尝试读取图片，失败返回路径，成功返回None"""
    try:
        img = cv2.imread(file_path)
        if img is None:
            return file_path
    except Exception:
        return file_path
    return None

def main():
    root_dir = '../Dataset'  # 当前目录（Dataset文件夹）
    subdirs = ['image', 'wrench', 'mixed_image']
    image_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    
    # 收集所有图片路径
    files_to_check = []
    for sub in subdirs:
        sub_path = os.path.join(root_dir, sub)
        if not os.path.isdir(sub_path):
            print(f"警告：文件夹 {sub_path} 不存在，跳过。")
            continue
        for root, _, files in os.walk(sub_path):
            for f in files:
                if f.lower().endswith(image_exts):
                    files_to_check.append(os.path.join(root, f))
    
    print(f"共找到 {len(files_to_check)} 张图片，开始检查...")
    
    corrupted = []
    total = len(files_to_check)
    for i, path in enumerate(files_to_check, 1):
        # 显示进度（每100张或每10%打印一次）
        if i % max(1, total // 10) == 0 or i == total:
            print(f"进度：{i}/{total} ({(i/total*100):.1f}%)")
        result = check_image(path)
        if result is not None:
            corrupted.append(result)
    
    if corrupted:
        print("\n损坏的图片文件如下：")
        for c in corrupted:
            print(c)
        print(f"\n总计 {len(corrupted)} 个文件损坏。")
    else:
        print("所有图片均可正常读取，没有损坏文件。")

if __name__ == '__main__':
    main()