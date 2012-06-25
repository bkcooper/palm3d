import sys
import serial
(xMov, yMov) = (float(sys.argv[-2]), float(sys.argv[-1]))
print "Moving by %0.2f, %0.2f"%(xMov, yMov)
ser = serial.Serial(4, 115200)
ser.write("r x=%0.2f y=%0.2f\r"%(xMov, yMov))      # write a string
ser.close()             # close port
