from pyAndorSDK3 import AndorSDK3
import numpy as np
import matplotlib.pyplot as plt

print("\nConnecting to camera...")
sdk3 = AndorSDK3()
cam = sdk3.GetCamera(0)

# Area Of Interest in the sCMOS chip.
# The chip has dimensions: 2160x2560 pixels**2.
cam.AOIHeight = 2160 # Height of the AOI
cam.AOIWidth = 2560 # Width of the AOI.
cam.AOILeft = 1 # First pixel from the left of the AOI, starts with 1.
cam.AOITop = 1 # First pixel from the top of the AOI, starts with 1.
# cam.ExposureTime = cam.min_ExposureTime
cam.ExposureTime = 0.00001 
cam.FrameRate = cam.max_FrameRate
cam.GateMode = "CW Off"
cam.MCPGain  = 4095
# queuing buffer
imgsize = cam.ImageSizeBytes
buf = np.empty((imgsize,), dtype='B')
cam.queue(buf, imgsize)
#cam.FrameCount =1
#cam.BitDepth='12 Bit'

# start
cam.AcquisitionStart()

# wait
acq = cam.wait_buffer(1000)

# stop
cam.AcquisitionStop()
cam.flush()

img = acq.raw_data

img2 = np.zeros((cam.AOIHeight * cam.AOIWidth), dtype=np.uint16)
pixel_processed = np.zeros((cam.AOIHeight,cam.AOIWidth), dtype=bool)

for k in range(0, (imgsize-3), 3):
    byte0 = np.uint16(img[k])
    byte1 = np.uint16(img[k+1])
    byte2 = np.uint16(img[k+2])
    
    idx_pxA = int(2 * k / 3)
    # print(idx_pxA)
    idx_pxB = int(2 * k / 3 + 1)
    # print(idx_pxB)
    
    valA =  np.uint16(byte0 << 4) + np.uint16(byte1 & 0x0F)
    valB = np.uint16(byte2 << 4) + np.uint16(byte1 >> 4)
    img2[idx_pxA] = valA
    img2[idx_pxB] = valB
    # pixel_processed[idx_pxA] = True
    # pixel_processed[idx_pxB] = True
    
img3 = np.reshape(img2, (cam.AOIHeight, cam.AOIWidth))
    
# img_len = len(img)

# img_prima = img[:(cam.AOIHeight*cam.AOIWidth)]

# img2 = np.reshape(img_prima,(cam.AOIHeight,cam.AOIWidth))
# img2_A = img2[:,:]
#np.save('lander2/laser810_CWOff_09ExpT_Filtro2.npy',img3)
#img2_B = img2[:,:,1]
plt.figure()
plt.imshow(img3)
plt.colorbar()
plt.show()
# plt.clim(95,120)

# np.savetxt("img2_A.txt",img2_A)

cam.close()

print("Done")
