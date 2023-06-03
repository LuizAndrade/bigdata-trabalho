import uvicorn
from fastapi import FastAPI

# Inicializando aplicação
app = FastAPI()


# Endpoint para checar se aplicação está respondendo
@app.get("/healthcheck")
def app_healthcheck():
    return {"message": True}


# Estrutura para executar rodando o código `python view.py`
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
