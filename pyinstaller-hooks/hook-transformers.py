# transformers 的 _LazyModule 机制在运行时通过 define_import_structure()
# 用 AST 解析 __init__.py 的 TYPE_CHECKING 块来构建延迟导入表。
# "pyz+py"：字节码打进 PYZ（保持加载速度），同时在 _internal 保留 .py 源文件
# 供 define_import_structure / open() 读取。
# 否则 AutoBackbone / AutoConfig 等通过 _LazyModule 加载子模块时会失败
# （如 VitDetBackbone 的 "Could not import module" 错误）。

from PyInstaller.utils.hooks import collect_submodules

# collect_submodules 已覆盖 transformers 全部子模块，无需手动补充。
# 保留 pyz+py 模式以确保 .py 源文件可用。
hiddenimports = collect_submodules("transformers")
module_collection_mode = "pyz+py"
