# torchvision 的模型类（如 VitDetBackbone）在 torch.load / pickle 反序列化时
# 需要 .py 源码才能定位类定义。"pyz+py"：字节码打进 PYZ（保持加载速度），
# 同时在 _internal 保留 .py 源文件供 pickle 使用。
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = collect_submodules("torchvision")
module_collection_mode = "pyz+py"
