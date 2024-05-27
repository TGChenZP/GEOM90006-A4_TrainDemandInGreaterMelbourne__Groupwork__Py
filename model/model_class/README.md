# GEOM90006-A4_TrainDemandInGreaterMelbourne__Groupwork__Py

## Folders
### Data (Git-ignored): 
Raw data downloaded using web scraping or from website
- `1270055001_sa2_2016_aust_shape`: SA2 Shapefile 2016 (68.1 MB)
- `1270055001_gccsa_2016_aust_shape`: GCCSA shapefile 2016 (29.9 MB)
- `SA2_2021_AUST_SHP_GDA2020`: SA2 shapefile 2021 (68.7 MB)
- `GCCSA_2021_AUST_SHP_GDA2020`: GCCSA shapefile 2021 (29.3 MB)
- `SA2-T02_Selected_Medians_and_Averages.csv`: 2021 Census by SA2 (124 KB)
- `SA2-P02_Selected_Medians_and_Averages-Census_2016.csv`: 2016 Census by SA2 (152 KB)
- `Metro Stops`: Locations of Metro ststions (104 KB)
- `POPULATION_2023_SA2_GDA2020`: Population density **until** 2023 (70.0 MB)
- `non-residential_facilities`: location of hospitals, schools, shopping centres and sport facilities (3.88 MB)
- `DEM_VIC`: Victoria DEM file (11.6 GB)

### Notebooks
- `DataScraping.ipynb`: Scrape non-residential facilities from websites 
- `proposal_map.ipynb`: Script for output map in proposal
- `preprocessing_xxxxx`: Preprocessing files (unify crs, geographical daggregation, clean unuseful attributes), output are stored in `output/preprocessing` folder.