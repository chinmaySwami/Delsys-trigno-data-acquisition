This code is an extension to an already existing repository on github. Following is the link to the repository:
https://github.com/axopy/pytrigno

Using pytrigno, the IMU data is read from Delsys Trigno Avanti sensors and is being plotted in realtime and saving the acquired data into a CSV file.
The code consists of two files.

Files and description: 

acquire_data.py : This program reads the data steam from the delsys trigno control utility and once keyboard interrupt is invoked, saves the acquired data into a csv file. Channels from 0 to 63 are being read. Can be used to read 144 channels by changing the channel range to 0 to 143 to read IMU data from 16 channels simultaineously. 

trigno_demo.py : This program plots the data being received from the delsys sensors in realttime. Currently I am only plotting the X, Y and Z axis of the accelerometer. A separate thread is created to just acquire the data and the main thread is used to plot the acquired data. Matplotlib's FuncAnimation is used for plotting.

Please note that this is for Delsys Trigno Avanti sensors and cannot be used with other sensors. Kindly refer to the delsys SDK documentation for modifying the pytrigno.py file as per the type of sensor being used.



