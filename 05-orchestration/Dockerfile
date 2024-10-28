# Usar una imagen base de Python
FROM python:3.10.12-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar los archivos del modelo, DictVectorizer y Pipfile
#COPY ["model1.bin", "dv.bin", "./"]
COPY ["model1.bin", "dv.bin", "Pipfile", "Pipfile.lock", "./"]

# Instalar pipenv
RUN pip install --no-cache-dir pipenv
#RUN pip install jsonify && \
#    pip install flask && \
#    pip install joblib && \
#    pip install requests && \
#    pip install scikit-learn==1.5.2

# Instalar las dependencias desde el Pipfile
RUN pipenv install --deploy --ignore-pipfile
#RUN pipenv install --deploy --system

# Establecer la variable de entorno para Flask
ENV FLASK_ENV=development
ENV FLASK_APP=app.py 

# Copiar el código de la aplicación
COPY app.py .

# Exponer el puerto en el que se ejecutará la aplicación
EXPOSE 5000

# Comando para ejecutar la aplicación
CMD ["pipenv", "run", "flask", "run", "--host=0.0.0.0", "--port=5000"]
#CMD ["pipenv", "run", "python", "app.py"]
#CMD ["python", "app.py"]
