# mobile_sam 的模型类在 torch.load 反序列化 checkpoint 时需要 .py 源码。
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = collect_submodules("mobile_sam")
module_collection_mode = "pyz+py"
