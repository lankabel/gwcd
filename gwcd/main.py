# this is the main py file

from gwcd.data.data import *
from gwcd.hlc.hlc import *


taxi = FlowData(table_name='aggregated_taxi', path='/home/sebastijan/project/gwcd/data')
taxi.get_data(geom_col = "flow", overwrite=False, limit=10000)
taxi.save_geom(overwrite=False)
taxi.make_graph(length_row_name='trip_length_km', overwrite=False)
taxi.save_graph(overwrite=False)

g = taxi.graph

algorithm = HLC(g, 10)
results = algorithm.run()
print("Threshold = %.6f" % algorithm.last_threshold)
print("D = %.6f" % algorithm.last_partition_density)

results_list = list(results)
len(results_list)

results_graph = assign_names(result=results_list, graph=g)




result_gpd = gpd.GeoDataFrame(list(map(lambda x:x.attributes(), list(g.es()))), geometry='flow')
result_gpd.to_file("results.gpkg", driver="GPKG")