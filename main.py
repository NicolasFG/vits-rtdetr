from ultralytics import RTDETR
import os
from roboflow import Roboflow
import locale
import torch
import yaml

locale.getpreferredencoding = lambda: "UTF-8"

ruta_actual = os.getcwd()

rf = Roboflow(api_key="VvcEs35JKM2WFJY30ijR")
project = rf.workspace("tesis-8euxs").project("tesis-yolo")
dataset = project.version(1).download("yolov8")

def update_settings_yaml(new_path):
    settings_path = os.path.expanduser('~\\AppData\\Roaming\\Ultralytics\\settings.yaml')

    if os.path.exists(settings_path):
        with open(settings_path, 'r') as file:
            settings = yaml.safe_load(file)

        settings['datasets_dir'] = new_path

        with open(settings_path, 'w') as file:
            yaml.dump(settings, file)
    else:
        settings = {
            'settings_version': '0.0.4',
            'datasets_dir': new_path
            # ... otros ajustes si son necesarios ...
        }
        os.makedirs(os.path.dirname(settings_path), exist_ok=True)
        with open(settings_path, 'w') as file:
            yaml.dump(settings, file)

def main():
    model = RTDETR('rtdetr-l.pt')
    model.info()

    nombre_directorio_actual = os.path.join(ruta_actual, 'Tesis-Yolo-1')
    nuevo_nombre_directorio = os.path.join(ruta_actual, 'datasets')
    
    update_settings_yaml(ruta_actual + '/datasets')

    if os.path.exists(nombre_directorio_actual):
        os.rename(nombre_directorio_actual, nuevo_nombre_directorio)
        print(f"Directorio renombrado a: {nuevo_nombre_directorio}")
    else:
        print(f"El directorio {nombre_directorio_actual} no existe.")

    # Cargar el archivo data.yaml
    ruta_data_yaml = os.path.join(ruta_actual, 'datasets/data.yaml')  # Asegúrate de que ruta_actual sea el directorio donde está data.yaml
    with open(ruta_data_yaml, 'r') as file:
        data_yaml_content = yaml.safe_load(file)

    # Actualizar las rutas de train y val
    data_yaml_content['train'] = ruta_actual + '/datasets/train/images'
    data_yaml_content['val'] = ruta_actual + '/datasets/valid/images'

    # Guardar el archivo data.yaml actualizado
    with open(ruta_data_yaml, 'w') as file:
        yaml.dump(data_yaml_content, file, sort_keys=False)

    model.train(data= ruta_actual + '/datasets/data.yaml', epochs=40, imgsz=416, batch=10, val=False)

    # Guardar el modelo en la carpeta "resultados"
    model_path = ruta_actual + '/resultados/model_trained.pt'
    torch.save(model.state_dict(), model_path)

main()
