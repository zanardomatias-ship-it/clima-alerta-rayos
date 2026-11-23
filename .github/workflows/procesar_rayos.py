
import json
import os
import re
import tempfile
from datetime import datetime, timedelta, timezone

import boto3
import numpy as np
import xarray as xr
from botocore import UNSIGNED
from botocore.config import Config

# CLIMA ALERTA - Rayos reales NOAA GOES-19
BUCKET = "noaa-goes19"
PRODUCTO = "GLM-L2-LCFA"

# Límites aproximados de Argentina.
LAT_MIN = -56.0
LAT_MAX = -21.0
LON_MIN = -74.0
LON_MAX = -53.0

MAX_ARCHIVOS = 10
MAX_RAYOS = 3000

s3 = boto3.client(
    "s3",
    config=Config(signature_version=UNSIGNED),
)


def fecha_archivo(nombre):
    """Extrae la hora de inicio del archivo GOES."""
    coincidencia = re.search(
        r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})",
        nombre,
    )

    if not coincidencia:
        return None

    anio, dia, hora, minuto, segundo = map(
        int, coincidencia.groups()
    )

    return (
        datetime(anio, 1, 1, tzinfo=timezone.utc)
        + timedelta(
            days=dia - 1,
            hours=hora,
            minutes=minuto,
            seconds=segundo,
        )
    )


def buscar_archivos():
    """Busca archivos recientes en el bucket público."""
    ahora = datetime.now(timezone.utc)
    encontrados = []

    # Consultar las últimas tres horas.
    for horas_atras in range(4):
        fecha = ahora - timedelta(hours=horas_atras)

        prefijo = (
            f"{PRODUCTO}/"
            f"{fecha.year}/"
            f"{fecha.timetuple().tm_yday:03d}/"
            f"{fecha.hour:02d}/"
        )

        paginador = s3.get_paginator("list_objects_v2")

        for pagina in paginador.paginate(
            Bucket=BUCKET,
            Prefix=prefijo,
        ):
            for archivo in pagina.get("Contents", []):
                clave = archivo["Key"]

                if clave.endswith(".nc"):
                    fecha_inicio = fecha_archivo(clave)

                    if fecha_inicio is not None:
                        encontrados.append(
                            (fecha_inicio, clave)
                        )

    encontrados.sort(reverse=True)

    return encontrados[:MAX_ARCHIVOS]


def procesar_archivo(clave, fecha_inicio):
    """Descarga un NetCDF y extrae las detecciones."""
    resultado = []

    with tempfile.TemporaryDirectory() as carpeta:
        ruta = os.path.join(carpeta, "rayos.nc")

        s3.download_file(BUCKET, clave, ruta)

        with xr.open_dataset(
            ruta,
            engine="netcdf4",
            decode_times=False,
        ) as datos:

            if (
                "flash_lat" not in datos
                or "flash_lon" not in datos
            ):
                print("Archivo sin coordenadas:", clave)
                return resultado

            latitudes = np.asarray(
                datos["flash_lat"].values
            ).ravel()

            longitudes = np.asarray(
                datos["flash_lon"].values
            ).ravel()

            for lat, lon in zip(latitudes, longitudes):
                lat = float(lat)
                lon = float(lon)

                if not (
                    np.isfinite(lat)
                    and np.isfinite(lon)
                ):
                    continue

                if not (
                    LAT_MIN <= lat <= LAT_MAX
                    and LON_MIN <= lon <= LON_MAX
                ):
                    continue

                resultado.append({
                    "lat": round(lat, 5),
                    "lon": round(lon, 5),
                    "horaArchivo": (
                        fecha_inicio.isoformat()
                    ),
                    "tipo": "deteccion_satelital",
                })

    return resultado


def main():
    print("CLIMA ALERTA - Procesando rayos GOES-19")

    archivos = buscar_archivos()

    print("Archivos encontrados:", len(archivos))

    if not archivos:
        raise RuntimeError(
            "No se encontraron archivos recientes. "
            "Se conserva el JSON anterior."
        )

    rayos = []

    for fecha, clave in archivos:
        try:
            print("Procesando:", clave)

            nuevos = procesar_archivo(
                clave,
                fecha,
            )

            rayos.extend(nuevos)

            if len(rayos) >= MAX_RAYOS:
                break

        except Exception as error:
            print(
                "No se pudo procesar:",
                clave,
                str(error),
            )

    rayos = rayos[:MAX_RAYOS]

    # No publicar un resultado vacío si todos
    # los archivos fallaron al procesarse.
    if not rayos:
        print(
            "No hay detecciones para Argentina "
            "en los archivos procesados."
        )

    salida = {
        "fuente": "NOAA GOES-19 GLM",
        "actualizadoEn": datetime.now(
            timezone.utc
        ).isoformat(),
        "region": "Argentina",
        "tipo": "actividad_electrica_satelital",
        "cantidad": len(rayos),
        "rayos": rayos,
    }

    with open(
        "rayos.json",
        "w",
        encoding="utf-8",
    ) as archivo:
        json.dump(
            salida,
            archivo,
            ensure_ascii=False,
            indent=2,
        )

    print("Detecciones publicadas:", len(rayos))
    print("Archivo rayos.json generado")


if __name__ == "__main__":
    main()
