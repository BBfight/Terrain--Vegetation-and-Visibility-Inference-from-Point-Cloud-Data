import os
import utm
import math
import subprocess


'''
### USAGE ###
Input
	- in_dir: directory where the classified(ground and non ground must be identified already) 'las' files are located
	- out_dir_ directory where the output will be stored
	- las_height_path: path of the 'las_height' tool
	- las_noise_path: path of the 'las_noise' tool
	the 'las tools suite' is downloadable at http://www.cs.unc.edu/~isenburg/lastools/
Output
	- the 'las' files denoised and with real height points will be stored in the out_dir folder
 
The classified LAS files must be named as "area_%i_classified" in order to be properly detected by the function
	
'''
def filterPoints(in_dir, out_dir, las_height_path, las_noise_path, areas):

	# removing noise
	for i in range(areas):
		inFile = in_dir + "area_%i_classified.las" % (i+1)
		subprocess.call('"' + las_noise_path + '"' + ' -i "' + inFile + '" -step_xy 4 -step_z 1 -remove_noise' +
						' -odir "' + out_dir + '"' + (' -o "area_%i_denoised.las"' % (i+1)),
						shell=True)
		print 'removed noise from area ' + str(i+1)

	# computing real height of points
	for i in range(areas):
		inFile = in_dir + "area_%i_denoised.las" % (i+1)
		subprocess.call('"' + las_height_path + '"' + ' -i "' + inFile + '" -replace_z -drop_below 0' +
						' -drop_above 30 -odir "' + out_dir + '"' + (' -o "area_%i_height.las"' % (i+1)),
						shell=True)
		print 'computed height of area ' + str(i+1)