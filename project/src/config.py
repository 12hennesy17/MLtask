import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Загружаем переменные окружения из файла .env
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_NAME = os.getenv("CONFIG_NAME", "base.yaml")

# --- Настройки API (FastAPI) ---
# Порт и режим автоперезагрузки для разработки
API_PORT = int(os.getenv("API_PORT", 8000))
API_RELOAD = os.getenv("API_RELOAD", "True").lower() == "true"

def load_config(config_name=CONFIG_NAME):
    config_path = PROJECT_ROOT / "configs" / config_name
    if not config_path.exists():
        raise FileNotFoundError(f"Конфиг не найден по пути: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# Загружаем сырые данные
_raw = load_config()

# --- 1. Константы и Параметры ---
RANDOM_STATE = _raw["general"]["random_state"]
TARGET = _raw["general"]["target"]

# --- 2. Основные пути ---
ARTIFACTS_DIR = PROJECT_ROOT / _raw["dirs"]["artifacts"]
DATA_DIR = PROJECT_ROOT / _raw["dirs"]["data_raw"]
PRODUCTION_MODELS_DIR = PROJECT_ROOT / _raw["dirs"]["models"]
TRAIN_PARAMS_DIR = PROJECT_ROOT / _raw["dirs"]["configs"]
LOG_DIR = PROJECT_ROOT / _raw["dirs"]["logs"]
# --- 3. Подпапки артефактов (Исследовательская среда) ---
_as = _raw["artifacts_subdirs"] # сокращение для удобства

EDA_DIR = ARTIFACTS_DIR / _as["eda"]
EDA_FIGURES_DIR = ARTIFACTS_DIR / _as["eda_figures"]
FEATURE_ENG_DIR = ARTIFACTS_DIR / _as["features"]
MODELING_DIR = ARTIFACTS_DIR / _as["modeling"]
MODELING_FIGURES_DIR = ARTIFACTS_DIR / _as["modeling_figures"]

# --- 4. Промышленная среда (CLI) ---
CLI_ARTIFACTS_DIR = ARTIFACTS_DIR / _as["cli_base"]
SUBMISSIONS_DIR = ARTIFACTS_DIR / _as["cli_submissions"]
CLI_REPORTS_DIR = ARTIFACTS_DIR / _as["cli_reports"]


# --- 5. Пути к файлам данных ---
TRAIN_DATA = DATA_DIR / _raw["files"]["train"]
TEST_DATA = DATA_DIR / _raw["files"]["test"]

# --- Инициализация структуры ---
# Список всех папок, которые должны существовать на диске
FOLDERS_TO_CREATE = [
    ARTIFACTS_DIR,
    DATA_DIR,
    EDA_DIR,
    EDA_FIGURES_DIR,
    FEATURE_ENG_DIR,
    LOG_DIR,
    MODELING_DIR,
    MODELING_FIGURES_DIR,
    PRODUCTION_MODELS_DIR,
    SUBMISSIONS_DIR,
    CLI_ARTIFACTS_DIR,
    CLI_REPORTS_DIR,
    TRAIN_PARAMS_DIR,
]

def create_project_structure():
    """Создает необходимые папки, если их еще нет"""
    for folder in FOLDERS_TO_CREATE:
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)
            print(f"📁 Создана папка: {folder.relative_to(PROJECT_ROOT)}")

if __name__ == "__main__":
    create_project_structure()