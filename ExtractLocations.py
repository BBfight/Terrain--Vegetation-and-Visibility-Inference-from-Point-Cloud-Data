from fastkml import kml
import csv
import utm

'''
Input:
	- locations_kml: trees locations in latitude and longitude expressed as kml palcemarks
Output:
	- the functions will store the locations converted to UTM in a csv located in the out_path folder

'''
def extractLocations(locations_kml, trees_locations):
    """Function to convert the trees locations expressed trough palcemarks on google Earth
       to UTM and to store the converted locations on a csv file
    """
    # extracting the .doc file from which generate the structure
    doc = file(locations_kml).read()
    k = kml.KML()
    k.from_string(doc)
    
    # Create the KML object to store the parsed result
    k = kml.KML()
    
    # Read in the KML string
    k.from_string(doc)
    
    # getting al the palcemarks in the kml
    features = list(list(k.features())[0].features())
    
    # iterate over placemarks and save longitude and latitude on a csv file
    with open(trees_locations, 'w') as points_locations:
        fieldnames = ['id', 'easting', 'northing']
        writer = csv.DictWriter(points_locations, fieldnames=fieldnames)
        writer.writeheader()
        for placemark in features:
            latitude = placemark.geometry.y
            longitude = placemark.geometry.x
            utm_coordinates = utm.from_latlon(latitude, longitude)
            name = placemark.name
            writer.writerow({'id': str(name), 'easting': str(utm_coordinates[0]), 'northing': str(utm_coordinates[1])})


