import geopandas as gpd
import sqlalchemy as db 



class Data(object):

    _password = 'Hu57ta59!'
    _username = 'postgres'

    def __init__(self, table_name:str):
        super().__init__()
        self.table_name = table_name

    _engine = db.create_engine(f'''postgresql://{_username}:{_password}@localhost/postgres''')
    _metadata = db.MetaData()
    _con = _engine.connect()
    # engine.table_names()
    
    def get_data(self):
        table = db.Table(self.table_name, self._metadata, autoload=True, autoload_with=self._engine)
        query = table.select()
        # query_result = self._con.execute(query)
        # query_result_set = query_result.fetchall()
        
        self.data = gpd.read_postgis(sql=query, con=self._con, geom_col='flow')
        

# taxi = Data('aggregated_taxi')
# taxi.get_data()