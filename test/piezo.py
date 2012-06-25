import sys
import serial
zPos = float(sys.argv[-1])
#print "Moving to %0.2f"%(zPos)
ser = serial.Serial(4, 115200)
ser.write("move z=%0.2f\r"%(zPos))      # write a string
ser.close()             # close port
