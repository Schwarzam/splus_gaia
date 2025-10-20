### To download GAIA data

```
python3 retrieve_files.py http://cdn.gea.esac.esa.int/Gaia/gdr3/gaia_source/ /home/hdcasa/gaia
```

### Download splus

Preencher usuário e senha do splus.cloud (registro é livre para todos)
E completar os Args faltantes no arquivo.

```
download_splus.py 
```


### Fazendo os matches dos dois por coordenada. 

Adaptar o comando abaixo para suas opcoes. 

```
python3 match.py --fields dr6_list.csv --gaia-dir /home/astrodados/gaia_dr3 --outdir /home/astrodados3/gustavo/splus_gaia_crowded/ --radius 1 --gaia-cols="ra,dec" --splus-template="/home/astrodados3/splus/idr6_final/main/{field}_dual.fits" --max-fields=3000
```


### Creating one big table

The notebook `concatenate_datasets.ipynb` will create a single table from the individual S-PLUS and GAIA datasets.