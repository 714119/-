from pathlib import Path


def get_project_root():
    """获取项目根目录"""
    return Path(__file__).parent.parent.parent


def get_config_path():
    """获取配置文件路径"""
    return get_project_root() / 'config.yaml'


def get_data_path(data_type='processed'):
    """获取数据路径"""
    import yaml
    config_path = get_config_path()
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    data_path = Path(config['data_paths'][data_type])
    if not data_path.is_absolute():
        data_path = get_project_root() / data_path
    return data_path