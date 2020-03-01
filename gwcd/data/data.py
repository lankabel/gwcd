import pathlib as pathlib

import geopandas as gpd
import sqlalchemy as db 
import igraph as ig
import math


class FlowData(object):

    _password = 'Hu57ta59!'
    _username = 'postgres'


    def __init__(self, table_name:str, path = "./data"):
        super().__init__()
        self.table_name = table_name
        self._path = path

    _engine = db.create_engine(f'''postgresql://{_username}:{_password}@localhost/postgres''')
    _metadata = db.MetaData()
    _con = _engine.connect()
    # engine.table_names()
    
    def get_data(self, overwrite=False, geom_col = "geom", limit=0):
        '''
        Gets the data from the database and stores it as GeoPandas DataFrame
        '''
        
        if self.gpkg_exists() and not overwrite:
            print(f'''File exists on path: {self._path}/{self.table_name}.gpkg. To overwrite use overwrite=True flag''')
        if self.gpkg_exists() and overwrite:
            self.data = gpd.read_file(f'''{self._path}/{self.table_name}.gpkg''')
            print(f'''Data loaded from: {self._path}/{self.table_name}.gpkg''')
        else:     
            table = db.Table(self.table_name, self._metadata, autoload=True, autoload_with=self._engine)
            if limit:
                query = table.select().limit(limit)
            else:                
                query = table.select()
            # query_result = self._con.execute(query)
            # query_result_set = query_result.fetchall()
            
            self.data:gpd.GeoDataFrame = gpd.read_postgis(sql=query, con=self._con, geom_col=geom_col)
            print(f'''Loaded data in the memory. Use FlowData.save_geom() to save it to file.''')

                    
    def gpkg_exists(self):
        return pathlib.Path(f'''{self._path}/{self.table_name}.gpkg''').is_file()
    
    def graph_exists(self):
        return pathlib.Path(f'''{self._path}/{self.table_name}.gmlz''').is_file()
        
    def save_geom(self, overwrite=False):
        """
        Saves GeoDataFrame to geopackage on defined path, creates all nested folders to define the path
        """
        if self.gpkg_exists() and not overwrite:
            print(f'''File exists on path: {self._path}/{self.table_name}.gpkg. To overwrite use overwrite=True flag.\nFile not overwritten.''')
        else:
            pathlib.Path(self._path).mkdir(parents=True, exist_ok=True)
            self.data.to_file(filename=f'''{self._path}/{self.table_name}.gpkg''', driver="GPKG")
            print(f'''File saved on path: {self._path}/{self.table_name}.gpkg''')
          
        
    def make_graph(self, overwrite=False, count_row_name:str = 'count', length_row_name:str='length'):
        
        self.data_cleaned = self.data[self.data[length_row_name]>0].copy()
        
        def distance_function(distance:gpd.GeoSeries):   
                return 1./(distance.apply)(lambda x: pow(x,2))
        
        
        self.data_cleaned['distance_function'] = distance_function(self.data_cleaned[length_row_name])   
        full_path = f'''{self._path}/{self.table_name}.gmlz'''
        if self.graph_exists() and not overwrite:
            self.graph = ig.Graph.Read_GraphMLz(full_path)
            print(f'''File exists on path: {full_path} . To overwrite use overwrite=True flag.\nLoading graph from file.''')
            pass
        else:
            # self.data_cleaned['weight'] = self.data_cleaned[count_row_name]
            self.data_cleaned['weight'] = self.data_cleaned[count_row_name] * self.data_cleaned['distance_function']
            
            self.graph = ig.Graph.TupleList(self.data_cleaned.itertuples(index=False), directed=True, edge_attrs="weight")
            for item in list(self.data):
                self.graph.es[item] = self.data[item]
            print(f'''Created graph in memory. Use FlowData.save_graph() to save it to file.''')
        
        
    def save_graph(self, overwrite=False):
        """
        Saves Graph to GMLZ on defined path, creates all nested folders to define the path
        """
        
        full_path = f'''{self._path}/{self.table_name}.gmlz'''
        
        if self.graph_exists() and not overwrite:
            print(f'''File exists on path: {full_path}. To overwrite use overwrite=True flag.\nFile not overwritten.''')
        else:
            pathlib.Path(self._path).mkdir(parents=True, exist_ok=True)
            self.graph.write_graphmlz(full_path)
            print(f'''File saved on path: {full_path}''')
        
        
# taxi = FlowData(table_name='aggregated_taxi', path='/home/sebastijan/project/gwcd/data')
# taxi.get_data(geom_col = "flow")
# taxi.save_geom()
# taxi.make_graph(length_row_name='trip_length_km')
# taxi.save_graph()


