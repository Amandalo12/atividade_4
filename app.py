#python -m streamlit run app.py
import streamlit as st
import tensorflow as tf
import numpy as np

# 1. Dicionário de dados simulado (Tema ENEM: Horas de estudo vs Nota de 0 a 10)
horas_estudo = np.array([1.0, 2.0, 4.0, 6.0, 8.0, 10.0], dtype=float)
notas_enem = np.array([3.0, 5.0, 6.5, 8.0, 9.0, 10.0], dtype=float)

# 2. Criando uma Rede Neural Simples (Regressão Linear)
modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

modelo.compile(optimizer='sgd', loss='mean_squared_error')
modelo.fit(horas_estudo, notas_enem, epochs=300, verbose=0)

# 3. Interface Web com Streamlit
st.title("📚 Simulador ENEM com Inteligência Artificial")
st.write("Descubra a nota estimada baseada nas suas horas de estudo diárias!")

# Entrada do usuário via slider
horas_usuario = st.slider("Quantas horas você estuda por dia?", 1.0, 12.0, 4.0)

# 4. Realizando a Previsão com o TensorFlow
nota_prevista = modelo.predict(np.array([horas_usuario]))[0][0]

# Exibindo o resultado na tela
st.success(f"Sua nota estimada no ENEM é: **{nota_prevista:.2f} / 10**")