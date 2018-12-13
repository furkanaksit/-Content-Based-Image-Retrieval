##
import numpy as np
import skimage.io as io
#from skimage import data, novice 
import matplotlib.pyplot as plt



##

def rgb2hsv(img):
    
    # use for converting rgb colorspace image to hsv colorspace image
    
    img_hsv = np.empty((img.shape))
    hsv_arr = np.zeros(360)
    divid = img.shape[0]*img.shape[1]
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            r = img[i,j,0]/255.0
            g = img[i,j,1]/255.0
            b = img[i,j,2]/255.0
            
            r, g, b = r/255.0, g/255.0, b/255.0
            
            mx = max(r, g, b)
            mn = min(r, g, b)
            df = mx-mn
            if mx == mn:
                h = 0
            elif mx == r:
                h = (60 * (((g-b)/df) % 6))
            elif mx == g:
                h = (60 * ((b-r)/df + 2)) 
            elif mx == b:
                h = (60 * ((r-g)/df + 4)) 
            if mx == 0:
                s = 0
            else:
                s = df/mx
            v = mx
            
            img_hsv[i,j,0] = h
            img_hsv[i,j,1] = s
            img_hsv[i,j,2] = v
            hsv_arr[int(h)] += 1/divid
    return img_hsv, hsv_arr
                              
def rgb2gray(img):

    # use for conveerting rgb image to grayscale image
    # in this context it is important for dealing with histograms
    # to many information would be more compex when it wouldnt do much difference
    
    img_g = np.zeros((img.shape[0],img.shape[1]))
    
    weights = np.array([0.2989, 0.5870, 0.1140])
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_g[i, j] = np.matmul(img[i, j, :],weights)
    return img_g        


       
def value(val_ar):
    
    # for calculating the decimal value of 8 bit list
    
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0
    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]
    return int(val)    

def get_pixel(center, px):
    
    # when calculating local binary pattern this module checks if the neighboor
    # pixel is higher or lower than our central pixel so we can choose if it is 
    # 0 or 1
    
    new_value = 0
    if px >= center:
        new_value = 1
    return new_value

def check_valid(ebit):
    
    # we can only allow if number of 0-1/1-0 transition is lower or equal to 2
    # this module counts the transition and flags it if it is valid
    res = 0
    check = 0
    for x in range(1,8):
        if ebit[x] - ebit[x-1] != 0:
            check += 1
    if check <= 2:
        res = 1
    return res    
      
    
def lbp_matrix(img):
    
    # this module makes 8-bit list for local binary pattern, checks the validity
    # of the lbp then makes the hsv coded image and lbp histogram
    
    lbp_arr = np.zeros(256)
    lbp_img = np.zeros((img.shape[0], img.shape[1]))
    img_g = rgb2gray(img)
    
    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            ebit = np.zeros(8)          
            ebit[0] = get_pixel(img_g[i,j], img_g[i-1, j+1])     # top_right
            ebit[1] = get_pixel(img_g[i,j], img_g[i, j+1])       # right
            ebit[2] = get_pixel(img_g[i,j], img_g[i+1, j+1])     # bottom_right
            ebit[3] = get_pixel(img_g[i,j], img_g[i+1, j])       # bottom
            ebit[4] = get_pixel(img_g[i,j], img_g[i+1, j-1])     # bottom_left
            ebit[5] = get_pixel(img_g[i,j], img_g[i, j-1])       # left
            ebit[6] = get_pixel(img_g[i,j], img_g[i-1, j-1])     # top_left
            ebit[7] = get_pixel(img_g[i,j], img_g[i-1, j])       # top
            if check_valid(ebit) == 1:
                lbp_img[i,j] = value(ebit)
                lbp_arr[value(ebit)] += 1 
    return lbp_arr, lbp_img

def lbp_hist(img):
    
    # to seperate lbp_arr from lbp_img, also can be used to display histogram 
    # and the image
    
    lbp_arr, lbp_img = lbp_matrix(img)        
    #plt.imshow(lbp_img)
    plt.show()
    #plt.hist(lbp_img)
    #plt.show()      
    return lbp_arr        

def hsv_hist(img):
    
    # to seperate hsv_arr from hsv_img, also can be used to display histogram 
    # and the image
    
    img_hsv, hsv_arr = rgb2hsv(img)
    #plt.imshow(img_hsv)
    #plt.hist(hsv_arr)
    #plt.show()
    return hsv_arr

def manhattan_distance_arr(x, y):
    
    # to calculate manhattan distance between two histograms, arrays
    
    sum = 0
    for i in range(len(x)):
        sum += abs(y[i] - x[i])
    return sum

def manhattan_distance_arr_2nd_degree(x1, x2, y1, y2):
    
    # to calculate manhattan distance between images that has two histograms
    
    sum = 0
    for i in range(len(x1)):
        sum += abs(x2[i] - x1[i]) 
    
    
    for i in range(len(y2)):
        sum += abs(y2[i] - y1[i])
    return sum

def sort_closest(closest):
    
    # matrix sorting algorithm of numpy was sorting such the columns are independent 
    # from each other but in our case row-0 is the distance and row-1 is the images
    # so we cannot think them as seperate 
    
    for i in range(len(closest[0,:])):
        min = closest[0,i]
        for j in range(i+1, len(closest[0,:])):
            if closest[0,j] < min:
                min = closest[0,j]
                temp = closest[0,i]
                closest[0,i] = closest[0,j]
                closest[0,j] = temp
                
                temp = closest[1,i]
                closest[1,i] = closest[1,j]
                closest[1,j] = temp
                

            

def five_closest(img, img_database):
    
    # module is used for finding five closest images for a given image but it may
    # have some different use too. at the end it compares the matrices
    
    closest_hsv = np.empty((2, 5), dtype=np.object) #closest[0,:] distances, closest[0,:] images
    closest_hsv[0, :] = img.shape[0]*img.shape[1]
    
    closest_lbp = np.empty((2, 5), dtype=np.object) #closest[0,:] distances, closest[0,:] images
    closest_lbp[0, :] = img.shape[0]*img.shape[1]
    
    closest_hsv_lbp = np.empty((2, 5), dtype=np.object) #closest[0,:] distances, closest[0,:] images
    closest_hsv_lbp[0, :] = img.shape[0]*img.shape[1]
    h = hsv_hist(img)
    l = lbp_hist(img)
    l_h = lbp_hist(img)
    
    for i in range(len(img_database)):

        sort_closest(closest_hsv)
        sort_closest(closest_lbp)
        sort_closest(closest_hsv_lbp)
        
        dist_hsv = manhattan_distance_arr(h, hsv_hist(img_database[i]))
        dist_lbp = manhattan_distance_arr(l, lbp_hist(img_database[i]))
        dist_hsv_lbp = manhattan_distance_arr(l_h, lbp_hist(img_database[i]))
        
        print(i , ' ' , end='')
        
        if dist_hsv < closest_hsv[0, 4]:
            closest_hsv[0, 4] = dist_hsv
            closest_hsv[1, 4] = img_database[i]
        
        if dist_lbp < closest_lbp[0, 4]:
                closest_lbp[0, 4] = dist_lbp
                closest_lbp[1, 4] = img_database[i]
        
        if dist_hsv_lbp < closest_hsv_lbp[0, 4]:
                closest_hsv_lbp[0, 4] = dist_hsv_lbp
                closest_hsv_lbp[1, 4] = img_database[i]
    
    return closest_hsv, closest_lbp, closest_hsv_lbp            







def main():
	img_database = io.ImageCollection('./*.jpg')
	#print(len(img_database))
	img = io.imread('./grass5.jpg')
	
		
	closest_hsv, closest_lbp, closest_hsv_lbp  = five_closest(img, img_database)
	plt.imshow(img)
	plt.title('actual image')
	plt.show()
    
	for i in range(len(closest_hsv[1,:])):
		plt.imshow(closest_hsv[1,i])
		plt.title("closest_hsv  distance = " + str(closest_hsv[0,i]))
		plt.show()
	for i in range(len(closest_lbp[1,:])):
		plt.imshow(closest_lbp[1,i])
		plt.title("closest_lbp  distance = " + str(closest_lbp[0,i]))
		plt.show()
	for i in range(len(closest_hsv_lbp[1,:])):
		plt.imshow(closest_hsv_lbp[1,i])
		plt.title("closest_hsv_lbp  distance = "+ str(closest_hsv_lbp[0,i]))
		plt.show()

main()




