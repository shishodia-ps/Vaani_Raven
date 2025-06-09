# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

block_cipher = None

# Define the main script
main_script = str(project_root / 'main.py')

# Collect all Python files in the project
python_files = []
for pattern in ['*.py', '**/*.py']:
    python_files.extend(project_root.glob(pattern))

# Convert to relative paths for PyInstaller
datas = []
for file_path in python_files:
    if file_path.name != 'main.py':
        relative_path = file_path.relative_to(project_root)
        datas.append((str(file_path), str(relative_path.parent)))

# Add configuration files
config_files = [
    (str(project_root / 'config' / 'system_config.yaml'), 'config'),
    (str(project_root / 'requirements.txt'), '.'),
]

# Add data directories
data_dirs = [
    (str(project_root / 'data'), 'data'),
    (str(project_root / 'logs'), 'logs'),
    (str(project_root / 'results'), 'results'),
]

# Combine all data files
all_datas = datas + config_files + data_dirs

a = Analysis(
    [main_script],
    pathex=[str(project_root)],
    binaries=[],
    datas=all_datas,
    hiddenimports=[
        'agents',
        'agents.pattern_agent',
        'agents.quant_agent', 
        'agents.sentiment_agent',
        'agents.risk_agent',
        'agents.execution_agent',
        'agents.meta_agent',
        'orchestrator',
        'orchestrator.runner',
        'data_pipeline',
        'data_pipeline.data_collector',
        'monitoring',
        'monitoring.dashboard',
        'testing',
        'testing.backtester',
        'training',
        'learning',
        'utils',
        'streamlit',
        'plotly',
        'torch',
        'transformers',
        'MetaTrader5',
        'stable_baselines3',
        'gymnasium',
        'sklearn',
        'scipy',
        'pandas',
        'numpy',
        'yaml',
        'asyncio',
        'logging',
        'pathlib',
        'datetime',
        'typing',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib',
        'IPython',
        'jupyter',
        'notebook',
        'pytest',
        'sphinx',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='VaaniRavenX',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
    version_file=None,
)
