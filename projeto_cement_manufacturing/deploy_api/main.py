from flask import Flask, render_template, request, redirect
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from class_concreto import Concreto
import pickle

# abrindo o modelo de ML
modelo = pickle.load(open('modelos/ml_modelo.sav', 'rb'))

# abrindo modelo de decodificao
decodificador = pickle.load(open('modelos/decodificador.sav', 'rb'))

# inciando aplicação
app = Flask(__name__)

# criando amostras bases
amostra1 = Concreto('Amostra padrão 1', 141.3, 212.0, 0.0, 203.5, 0.0, 971.8, 748.6, 28, 29.89) # 29.89
amostra2 = Concreto('Amostra padrão 2', 250.0, 0.0, 95.7, 187.4, 5.5, 956.9, 861.2, 28, 29.22) # 29.22

# lista
lista = [amostra1, amostra2]


@app.route('/')
def index():
    return render_template('lista.html', titulo='Modelo do fck', amostras=lista)


@app.route('/criar', methods=['POST'])
def fkc():
    # pegando dados do formulario
    nome = request.form['nome']
    cimento = float(request.form['cimento'])
    escoria = float(request.form['escoria'])
    cinzas = float(request.form['cinzas'])
    agua = float(request.form['agua'])
    plast = float(request.form['plast'])
    graudo = float(request.form['graudo'])
    miudo = float(request.form['miudo'])
    idade = float(request.form['idade'])

    # criando array
    arr = np.array([cimento, escoria, cinzas, agua, plast, graudo, miudo, idade]).reshape(1, -1)
    values = decodificador.transform(arr)

    # aplicando no modelo
    model = modelo.predict(values)

    # criando uma amostra
    amostra = Concreto(nome, cimento, escoria, cinzas, agua, plast, graudo, miudo, idade, fck=round(model[0], 2))

    # fixando na lista
    lista.append(amostra)
    return redirect('/')


@app.route('/duvidas')
def duvidas():
    return render_template('duvidas.html', titulo='Dúvidas')


app.run(debug=True)



