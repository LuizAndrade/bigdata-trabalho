import uvicorn
from fastapi import FastAPI
from spark import start_spark, train_data

# Inicializando aplicação
app = FastAPI()


# Endpoint para checar se aplicação está respondendo
@app.get("/spark")
def app_spark():
    spark_context = start_spark()
    train_data(spark_context)
    return {"message": True}


# Endpoint para checar se aplicação está respondendo
@app.get("/healthcheck")
def app_healthcheck():
    return {"message": True}


# Estrutura para executar rodando o código `python view.py`
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
