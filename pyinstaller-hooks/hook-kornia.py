# kornia 在 import 时对部分函数执行 torch.jit.script，inspect 需要能读到 .py 源码。
# "pyz+py"：字节码打进 PYZ（保持加载速度），同时在 _internal 保留 .py 源文件供 inspect 使用。
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = collect_submodules("kornia")
module_collection_mode = "pyz+py"
