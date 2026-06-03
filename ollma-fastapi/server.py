from fastapi import FastAPI


app=FastAPI()


@app.get('/')

def read_root():
    return {"Hello":"World"}


@app.get('/contact')

def read_root():
    return {"email":"rohit@gmail.com"}