import runpy

def test_etapa3_runs_without_errors():
    # Executa o script inteiro como se fosse "python gerar_graficos_etapa3.py"
    runpy.run_path("scripts/gerar_graficos_etapa3.py", run_name="__main__")
