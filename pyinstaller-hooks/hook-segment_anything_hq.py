# segment_anything_hq 的模型类在 torch.load 反序列化 checkpoint 时需要 .py 源码。
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = collect_submodules("segment_anything_hq")
module_collection_mode = "pyz+py"
