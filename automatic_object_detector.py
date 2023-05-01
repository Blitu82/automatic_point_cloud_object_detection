import numpy as np
import pandas as pd
import geopandas as gpd
import open3d as o3d
from shapely.geometry import box
from shapely.ops import unary_union
import matplotlib.pyplot as plt
from IPython.display import display
import folium

# Define input and output files
INPUT_FILE = 'wreck_2.xyz'
OUTPUT_FILE = 'wreck_2.html'
OUTPUT_TABLEAU_1 = 'polygon_wreck_2.shp'
OUTPUT_TABLEAU_2 = 'points_wreck_2.geojson'
OUTPUT_TABLEAU_3 = 'stats_wreck_2.geojson'

# Define parameters
GRID_SIZE = 1
THRESHOLD = 0.2
BUFFER_DISTANCE = 1

def format_file(file_raw, file_formatted):
    """
    Reads in a raw file of type `.xyz` and creates a formatted version of the file, with columns separated by whitespace.

    Args:
        file_raw (str): The name of the input `.xyz` file.
        file_formatted (str): The name of the output formatted file.

    Returns:
        None: This function only writes the formatted file to disk.
    """
    with open(file_raw, 'r') as file:
        lines = file.readlines()
        column1 = []
        column2 = []
        column3 = []
        for line in lines:
            columns = line.strip().split(' ')
            column1.append(columns[0])
            column2.append(columns[1])
            column3.append(columns[2])

        with open(file_formatted, 'w') as output_file:
            for i in range(len(column1)):
                output_file.write(f"{column2[i]} {column1[i]} {column3[i]}\n")
        print("Data has been saved to ", file_formatted, " file.")
               
def load_print(file):
    """
    Loads a point cloud file from disk using open3d and numpy. Prints the point cloud array.

    Args:
        file (str): The name of the input point cloud file.

    Returns:
        tuple: A tuple containing a numpy array and a point cloud object.
    """
    pcd = o3d.io.read_point_cloud(file)
    print(pcd)
    pc = np.genfromtxt(file, dtype=None)
    np.set_printoptions(suppress=True, precision=3)
    print(pc)
    return pc, pcd

def create_point_geodataframe(pc):
    """
    Takes in a numpy array of points and creates a Pandas GeoDataFrame.

    Args:
        pc (np.ndarray): A numpy array of points.

    Returns:
        gpd.geodataframe.GeoDataFrame: A GeoDataFrame containing the input points.
    """
    df = pd.DataFrame(pc, columns=['x', 'y','z'])
    geometry = gpd.points_from_xy(df['x'], df['y'])
    point_gdf = gpd.GeoDataFrame(df, crs='epsg:32619', geometry=geometry)
    point_gdf = point_gdf.drop(['x', 'y'], axis=1)
    print('Point Cloud loaded to a Pandas GeoDataFrame.')
    return point_gdf

def plot_point_geodataframe(point_gdf):
    """
    Plots a point cloud GeoDataFrame.

    Args:
        point_gdf (GeoDataFrame): GeoDataFrame containing point cloud data.

    Returns:
        None.
    """
    point_gdf.plot(marker='o', column='z', cmap='terrain', markersize=20)
    plt.xticks(rotation=90)
    plt.title('Point Cloud')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.ticklabel_format(useOffset=False, style='plain')
    plt.show()

def create_grid_geodataframe(point_gdf, grid_size):
    """
    Creates a regular grid of squares based on the bounding box of a point cloud GeoDataFrame.

    Args:
        point_gdf (GeoDataFrame): GeoDataFrame containing point cloud data.
        grid_size (float): Size of the grid cells.

    Returns:
        grid_gdf (GeoDataFrame): GeoDataFrame containing the regular grid of squares.
    """
    min_x, min_y, max_x, max_y = point_gdf.geometry.total_bounds
    grid_x = np.arange(min_x - 0.8, max_x + 0.8, grid_size)
    grid_y = np.arange(min_y - 0.8, max_y + 0.8, grid_size)
    geometry=[box(x, y, x + grid_size, y + grid_size) for x in grid_x for y in grid_y]
    grid_gdf = gpd.GeoDataFrame(crs='epsg:32619', geometry=geometry)
    grid_gdf['cell_id'] = grid_gdf.index
    print('Grid loaded to a Pandas GeoDataFrame.')
    return grid_gdf

def point_grid_join(point_gdf, grid_gdf):
    """
    Joins a point cloud GeoDataFrame and a regular grid GeoDataFrame based on their spatial relationship.

    Args:
        point_gdf (GeoDataFrame): GeoDataFrame containing point cloud data.
        grid_gdf (GeoDataFrame): GeoDataFrame containing the regular grid of squares.

    Returns:
        point_gdf (GeoDataFrame): GeoDataFrame containing the point cloud data with an added cell_id column.
    """
    point_gdf = gpd.sjoin(point_gdf, grid_gdf, how='left', predicate='within')
    point_gdf = point_gdf[['z', 'geometry', 'cell_id']]
    print('Point and Grid geodataframes joined together.')
    return point_gdf

def plot_point_grid(point_gdf, grid_gdf):
    """
    Plots a regular grid and a point cloud on the same plot.

    Args:
        point_gdf (GeoDataFrame): GeoDataFrame containing point cloud data with a cell_id column.
        grid_gdf (GeoDataFrame): GeoDataFrame containing the regular grid of squares.

    Returns:
        None.
    """
    fig, ax = plt.subplots(figsize=(20,20))
    grid_gdf.plot(ax=ax, color='none', edgecolor='black', linewidth=1)
    # for index, row in grid_gdf.iterrows():
        # ax.annotate(text=row['cell_id'], xy=row['geometry'].centroid.coords[0], ha='center', va='center')
    point_gdf.plot(ax=ax, marker='o', column='z', cmap='terrain', markersize=20,  legend=True)
    plt.title('Point Cloud and Grid')
    plt.xlabel('X')
    plt.ylabel('Y')
    ax.ticklabel_format(useOffset=False, style='plain')
    plt.xticks(rotation=90)
    plt.show()

def create_stats_geodataframe(point_gdf, grid_gdf):
    """
    Computes statistics for the point geodataframe and merges them with the grid geodataframe.

    Args:
    - point_gdf (GeoDataFrame): geodataframe with point data and cell_id column
    - grid_gdf (GeoDataFrame): geodataframe with polygon data and cell_id column

    Returns:
    - merged_gdf (GeoDataFrame): geodataframe with polygon data, cell_id column, and computed statistics
    """
    stats_df = point_gdf.groupby('cell_id').agg({'z': ['mean', 'min', 'max','std', 'count']})
    stats_df.columns = ['_'.join(col).strip() for col in stats_df.columns.values]
    stats_df = stats_df.reset_index()
    stats_gdf1 = gpd.GeoDataFrame(stats_df)
    stats_gdf = pd.merge(grid_gdf, stats_gdf1, on='cell_id', how='left')
    return stats_gdf

def plot_stats_geodataframe(stats_gdf):
    """
    Plots the merged geodataframe with the standard deviation per cell.

    Args:
    - merged_gdf (GeoDataFrame): geodataframe with polygon data, cell_id column, and computed statistics
    
    Returns: None
    """
    fig, ax = plt.subplots(figsize=(20, 20))
    # for idx, centroid in enumerate(stats_gdf.centroid):
    #     ax.text(centroid.x, centroid.y, f'{idx}', ha='center', va='center', fontsize=5)
    stats_gdf.plot(ax=ax,color='none', edgecolor='black', linewidth=1)
    stats_gdf.plot(ax=ax, column='z_std', legend=True)
    plt.title('Standard Deviation per cell')
    plt.xlabel('X')
    plt.ylabel('Y')
    ax.ticklabel_format(useOffset=False, style='plain')
    plt.xticks(rotation=90)
    plt.show()

def object_detector(stats_gdf, point_gdf, threshold, buffer_distance):
    """
    Extracts the points within a bounding polygon created from cells with standard deviation above a threshold.

    Args:
    - merged_gdf (GeoDataFrame): geodataframe with polygon data, cell_id column, and computed statistics
    - point_gdf (GeoDataFrame): geodataframe with point data and cell_id column
    - threshold (float): threshold for standard deviation of cell z values
    - buffer_distance (float): buffer distance around bounding polygon

    Returns:
    - bounding_polygon (GeoDataFrame): geodataframe with bounding polygon geometry
    - points_within_polygon (GeoDataFrame): geodataframe with point data within bounding polygon
    """
    selected_grid = stats_gdf[stats_gdf.z_std > threshold]
    cells = selected_grid['cell_id'].to_list()
    selected_points = point_gdf[point_gdf.cell_id.isin(cells)]

    # Create a polygon for each group of points above the threshold
    grouped_points = selected_points.groupby('cell_id')
    polygons = [grouped_points.get_group(cell).unary_union.convex_hull for cell in grouped_points.groups]

    # Combine all polygons into a single MultiPolygon
    multi_polygon = unary_union(polygons)

    # Create a buffer around the MultiPolygon
    buffered_polygon = multi_polygon.buffer(buffer_distance)

    # Create a GeoDataFrame for the bounding polygon
    bounding_polygon = gpd.GeoDataFrame({'geometry': [buffered_polygon]}, crs='epsg:32619')

    # Find the centroid of the bounding polygon
    centroid = buffered_polygon.centroid

    # Find the points within the bounding polygon
    points_within_polygon = gpd.sjoin(point_gdf, bounding_polygon, predicate='within')
    points_within_polygon = gpd.GeoDataFrame(points_within_polygon, geometry='geometry')
    return bounding_polygon, points_within_polygon

def plot_object(bounding_polygon, points_within_polygon, stats_gdf):
    """
    Plot a map with a polygon boundary and points within the polygon, using matplotlib.
    Args:
    - bounding_polygon: a geopandas GeoSeries containing a polygon boundary
    - points_within_polygon: a geopandas GeoDataFrame containing points within the polygon
    - merged_gdf: a geopandas GeoDataFrame containing merged data

    Returns: None
    """
    fig, ax = plt.subplots(figsize=(20, 20))
    # for idx, centroid in enumerate(merged_gdf.centroid):
    #     ax.text(centroid.x, centroid.y, f'{idx}', ha='center', va='center', fontsize=5)
    stats_gdf.plot(ax=ax,color='none', edgecolor='black', linewidth=1)
    # filtered_grid.plot(ax=ax, column='z_std', legend=True)
    points_within_polygon.plot(ax=ax, marker='o', column='z', cmap='terrain', markersize=20,  legend=True)
    bounding_polygon.plot(ax=ax,color='none', edgecolor='red', linewidth=2)
    bounding_polygon.centroid.plot(ax=ax, marker='x', color='red', markersize=100)
    plt.title('Extracted points')
    plt.xlabel('X')
    plt.ylabel('Y')
    ax.ticklabel_format(useOffset=False, style='plain')
    plt.xticks(rotation=90)
    plt.show()

def interactive_map(bounding_polygon, points_within_polygon, output_file):
    """
    Create an interactive map with a polygon boundary and points within the polygon, using folium.
    Args:
    - bounding_polygon: a geopandas GeoSeries containing a polygon boundary.
    - points_within_polygon: a geopandas GeoDataFrame containing points within the polygon.
    - output_file: a string specifying the name and location of the output file.

    Returns: None
    """
    m = folium.Map(location=[2.5888, -34.5171], zoom_start=19, max_zoom=100)
    bounding_polygon.explore(
        m=m,
        style_kwds=dict(color="red",fillOpacity=0),
        name="polygon"
    )
    points_within_polygon.explore(
        m=m,
        column="z",
        cmap="rainbow", 
        popup=True,
        tooltip="z",
        name="points",
        legend=True,
    )
    folium.LayerControl().add_to(m)
    m.add_child(folium.LatLngPopup())
    m.save(output_file)

def tableau_export(bounding_polygon, points_within_polygon, stats_gdf, output_tableau_1, output_tableau_2, output_tableau_3):
    """
    Export geospatial data to files for use in Tableau.
    Args:
    - bounding_polygon: a geopandas GeoSeries containing a polygon boundary
    - points_within_polygon: a geopandas GeoDataFrame containing points within the polygon
    - stats_gdf: a geopandas GeoDataFrame containing statistics data
    - output_tableau_1: a string specifying the name and location of the output file for bounding_polygon
    - output_tableau_2: a string specifying the name and location of the output file for points_within_polygon
    - output_tableau_3: a string specifying the name and location of the output file for stats_gdf

    Returns: None
    """
    bounding_polygon.to_file(output_tableau_1)
    points_within_polygon.to_file(output_tableau_2, driver='GeoJSON')
    stats_gdf.to_file(output_tableau_3, driver='GeoJSON')

def main():
    pc, pcd = load_print(INPUT_FILE)
    point_gdf = create_point_geodataframe(pc)
    grid_gdf = create_grid_geodataframe(point_gdf, GRID_SIZE)
    point_gdf = point_grid_join(point_gdf, grid_gdf)
    plot_point_grid(point_gdf, grid_gdf)
    stats_gdf = create_stats_geodataframe(point_gdf, grid_gdf)
    plot_stats_geodataframe(stats_gdf)
    bounding_polygon, points_within_polygon = object_detector(stats_gdf, point_gdf, THRESHOLD, BUFFER_DISTANCE)
    plot_object(bounding_polygon, points_within_polygon, stats_gdf)
    interactive_map(bounding_polygon, points_within_polygon, OUTPUT_FILE)
    tableau_export(bounding_polygon, points_within_polygon, stats_gdf, OUTPUT_TABLEAU_1, OUTPUT_TABLEAU_2, OUTPUT_TABLEAU_3)

if __name__ == '__main__':
    main()