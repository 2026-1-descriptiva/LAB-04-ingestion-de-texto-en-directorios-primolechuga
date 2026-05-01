# pylint: disable=import-outside-toplevel
# pylint: disable=line-too-long
# flake8: noqa
"""
Escriba el codigo que ejecute la accion solicitada en cada pregunta.
"""

import os
import zipfile

import pandas as pd  # type: ignore


def pregunta_01():
    """
    La información requerida para este laboratio esta almacenada en el
    archivo "files/input.zip" ubicado en la carpeta raíz.
    Descomprima este archivo.

    Como resultado se creara la carpeta "input" en la raiz del
    repositorio, la cual contiene la siguiente estructura de archivos:


    ```
    train/
        negative/
            0000.txt
            0001.txt
            ...
        positive/
            0000.txt
            0001.txt
            ...
        neutral/
            0000.txt
            0001.txt
            ...
    test/
        negative/
            0000.txt
            0001.txt
            ...
        positive/
            0000.txt
            0001.txt
            ...
        neutral/
            0000.txt
            0001.txt
            ...
    ```

    A partir de esta informacion escriba el código que permita generar
    dos archivos llamados "train_dataset.csv" y "test_dataset.csv". Estos
    archivos deben estar ubicados en la carpeta "output" ubicada en la raiz
    del repositorio.

    Estos archivos deben tener la siguiente estructura:

    * phrase: Texto de la frase. hay una frase por cada archivo de texto.
    * sentiment: Sentimiento de la frase. Puede ser "positive", "negative"
      o "neutral". Este corresponde al nombre del directorio donde se
      encuentra ubicado el archivo.

    Cada archivo tendria una estructura similar a la siguiente:

    ```
    |    | phrase                                                                                                                                                                 | target   |
    |---:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------|
    |  0 | Cardona slowed her vehicle , turned around and returned to the intersection , where she called 911                                                                     | neutral  |
    |  1 | Market data and analytics are derived from primary and secondary research                                                                                              | neutral  |
    |  2 | Exel is headquartered in Mantyharju in Finland                                                                                                                         | neutral  |
    |  3 | Both operating profit and net sales for the three-month period increased , respectively from EUR16 .0 m and EUR139m , as compared to the corresponding quarter in 2006 | positive |
    |  4 | Tampere Science Parks is a Finnish company that owns , leases and builds office properties and it specialises in facilities for technology-oriented businesses         | neutral  |
    ```


    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    files_dir = os.path.join(repo_root, "files")
    zip_path = os.path.join(files_dir, "input.zip")
    input_dir = os.path.join(files_dir, "input")
    output_dir = os.path.join(files_dir, "output")

    if not os.path.isdir(os.path.join(input_dir, "train")):
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(files_dir)

    if os.path.isdir(os.path.join(input_dir, "input")):
        input_dir = os.path.join(input_dir, "input")

    os.makedirs(output_dir, exist_ok=True)

    def build_dataset(split_name: str):
        records = []
        split_path = os.path.join(input_dir, split_name)

        if not os.path.isdir(split_path):
            return pd.DataFrame(records, columns=["phrase", "target"])

        for sentiment in sorted(os.listdir(split_path)):
            sentiment_path = os.path.join(split_path, sentiment)
            if not os.path.isdir(sentiment_path):
                continue

            for filename in sorted(os.listdir(sentiment_path)):
                if not filename.lower().endswith(".txt"):
                    continue

                file_path = os.path.join(sentiment_path, filename)
                with open(file_path, "r", encoding="utf-8") as handle:
                    phrase = handle.read().strip()

                records.append({"phrase": phrase, "target": sentiment})

        return pd.DataFrame(records, columns=["phrase", "target"])

    train_df = build_dataset("train")
    test_df = build_dataset("test")

    train_df.to_csv(os.path.join(output_dir, "train_dataset.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test_dataset.csv"), index=False)
