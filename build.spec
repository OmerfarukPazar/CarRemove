# -*- mode: python -*-
import torch
import inspect
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None
excluded_modules = ['torch.distributions']

a = Analysis(['predict.py'],
             pathex=['D:\\Tamamlanan\\CarRemove'],
             binaries=[],
             datas=[
                ('models' , 'models'),
                ('panoComplete.ui', '.'),
                ('configs' , 'configs'),
                ('build','build'),
                ('big-lama','big-lama'),
                 *collect_data_files("torch", include_py_files=True),
                 *collect_data_files("kornia", include_py_files=True),
                 *collect_data_files("saicinpainting", include_py_files=True),
                 *collect_data_files("torchvision", include_py_files=True)
             ],
             hiddenimports=[
                 'pkg_resources.py2_warn',
                 'PIL',
                 'PIL.Image',
                 'PIL.ImageTk',
                 'cv2',
                 'numpy',
                 'tkinter',
                 'tkinter.filedialog',
                 'tkinter.messagebox',
                 'tkinter.ttk',
                 'torch',
                 'torch.nn',
                 'torch.jit',
                 'torchvision',
                 'kornia',
                 'kornia.color',
                 'kornia.geometry',
                 'kornia.filters',
                 'saicinpainting'
             ],
             hookspath=[],
             runtime_hooks=[],
             excludes=excluded_modules,
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)

a.datas += [ ('icon.ico', '.\\icon.ico', 'DATA')]

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='CarRemove',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=False)

coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='CarRemove')