# run_all.ps1

# Permite la ejecución de scripts para esta sesión.
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned

# Inicia el servidor FastAPI (uvicorn) en una nueva ventana, activando el entorno virtual.
Start-Process powershell -ArgumentList '-NoExit', '-Command', '. .\app\Scripts\activate; uvicorn api:app --reload'

# Inicia la aplicación Streamlit en otra nueva ventana, activando también el entorno virtual.
Start-Process powershell -ArgumentList '-NoExit', '-Command', '. .\app\Scripts\activate; streamlit run app.py'
