import pytest
from typer.testing import CliRunner
from src.cli import app
import pandas as pd


runner = CliRunner()

@pytest.fixture
def sample_csv(tmp_path):
    """Создает временный CSV файл для тестов."""
    d = tmp_path / "data"
    d.mkdir()
    p = d / "train.csv"
    df = pd.DataFrame({
    'Id': range(30),
    'SalePrice': [float(100000 + i*3000) for i in range(30)],
    'GrLivArea': [float(1000 + i*15) for i in range(30)],
    'LotArea': [float(5000 + i*200) for i in range(30)],
    'Feature1': [float(i * 0.5) for i in range(30)],
    'Feature4': [float(300 + i*200) for i in range(30)],
    'Feature5': [float(900 + i*400) for i in range(30)],
    })
    df.to_csv(p, index=False)
    return p

def test_overview_command(sample_csv):
    """Проверяем, что команда overview отрабатывает без ошибок."""
    result = runner.invoke(app, ["overview", str(sample_csv)])
    assert result.exit_code == 0
    assert "Строк: 30" in result.output
    assert "Столбцов: 7" in result.output

def test_train_command_fail_no_config(sample_csv):
    """Проверяем, что train падает, если нет JSON конфига."""
    # Передаем несуществующий путь к конфигу
    result = runner.invoke(app, ["train", str(sample_csv), "--config-path", "non_existent.json"])
    assert result.exit_code != 0  # Должен быть Exit или ошибка

def test_report_command(sample_csv, tmp_path):
    """Проверяем генерацию отчета."""
    out_dir = tmp_path / "reports"
    result = runner.invoke(app, ["report", str(sample_csv), "--out-dir", str(out_dir)])
    
    assert result.exit_code == 0
    assert (out_dir / "report.md").exists()
    assert (out_dir / "summary.csv").exists()

def test_healthcheck_fail_no_model():
    """Проверяем healthcheck при отсутствии модели."""
    # Передаем путь к несуществующей модели
    result = runner.invoke(app, ["healthcheck", "--model-path", "no_model.joblib"])
    assert result.exit_code != 0
    assert "Модель не найдена" in result.output 