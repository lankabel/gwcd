import pathlib as pathlib

import geopandas as gpd
import sqlalchemy as db 
import igraph as ig



class FlowData(object):

    _password = 'Hu57ta59!'
    _username = 'postgres'
    _path = "./data"

    def __init__(self, table_name:str, path = "./data"):
        super().__init__()
        self.table_name = table_name
        self._path = path

    _engine = db.create_engine(f'''postgresql://{_username}:{_password}@localhost/postgres''')
    _metadata = db.MetaData()
    _con = _engine.connect()
    # engine.table_names()
    
    def get_data(self, overwrite=False, geom_col = "geom"):
        '''
        Gets the data from the database and stores it as GeoPandas DataFrame
        '''
        
        if self.gpkg_exists() and not overwrite:
            print(f'''File exits on path: {self._path}/{self.table_name}.gpkg. To overwrite use overwrite=True flag''')
        if self.gpkg_exists() and overwrite:
            self.data = gpd.read_file(f'''{self._path}/{self.table_name}.gpkg''')
            print(f'''Data loaded from: {self._path}/{self.table_name}.gpkg''')
        else:     
            table = db.Table(self.table_name, self._metadata, autoload=True, autoload_with=self._engine)
            query = table.select()
            # query_result = self._con.execute(query)
            # query_result_set = query_result.fetchall()
            
            self.data:gpd.GeoDataFrame = gpd.read_postgis(sql=query, con=self._con, geom_col=geom_col)
        
    def gpkg_exists(self, path=_path):
        return pathlib.Path(f'''{path}/{self.table_name}.gpkg''').is_file()
        
        
    def save_geom(self, path=_path, overwrite=False):
        """
        Saves GeoDataFrame to geopackage on defined path, creates all nested folders to define the path
        """
        if self.gpkg_exists() and not overwrite:
            print(f'''File exits on path: {self._path}/{self.table_name}.gpkg. To overwrite use overwrite=True flag.\n File not saved.''')
        else:
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)
            self.data.to_file(filename=f'''{path}/{self.table_name}.gpkg''', driver="GPKG")
        
        
    def make_graph(self):
        pass
        
        
def test():
    taxi = FlowData(table_name='aggregated_taxi', path='/home/sebastijan/project/gwcd/data')
    taxi.get_data(geom_col = "flow")
    taxi.save_geom()