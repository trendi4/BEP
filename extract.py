from PIL import Image
import numpy as np
import os
import time
import shutil
import sys
import glob

def main():
    print("Starting program:\n")
    start_time = time.time()
    multiplier = 1.5
    version = 'prn' 

    txtfiles = []
    for file in glob.glob("tif_stacks/*.tif"):
        file = file[11:-4]
        txtfiles.append(file)
    #tif_to_jpg_op = 1  # 0 for no conversion, 1 for conversion
    #crop_jpg_op = 1  # 0 for no ccropping, 1 for cropping
    if "tif_to_jpg" in str(sys.argv):
        tif_to_jpg(txtfiles)
        crop(txtfiles)
    #if "crop_jpg" in str(sys.argv):
        
    if str(sys.argv[-1][0]).isdigit():
        multiplier = float(sys.argv[-1])    
    if "yolov4" in str(sys.argv):
        version='yolov4' 
    if "pan" in str(sys.argv):
        version='pan' 
    extract_patches(txtfiles,multiplier,version)
    end_time = time.time()
    
    print("\nProgram ended. Total run time: " +
          str(round(end_time-start_time, 4)) + " seconds")

#    names=["GEN2LFIM_16_Cancer_F6_F489_A1252", 'GEN2LFIM_13_Cancer_F3_F569_A376', 'DDP_12_002_Squamous_F5_F955_A3128','DDP_10_005_NS2_VLE_F5_F642_A2173', 'DDP_05_008_S2_VLE_F4_F940_A3531', 'AMC_05_NDBE_F465_342mm_2300', 'AMC_04_HGD_F545_303mm_2130', 'AMC_03_LGD_F990_361mm_1630','AMC_02_NDBE_F630_361mm_1900', 'AMC_01_NDBE_F803_370mm_1800'] 



def extract_patches(names,multiplier=1.5, version='prn'):
 #names=["GEN2LFIM_16_Cancer_F6_F489_A1252", 'GEN2LFIM_13_Cancer_F3_F569_A376', 'DDP_12_002_Squamous_F5_F955_A3128','DDP_10_005_NS2_VLE_F5_F642_A2173', 'DDP_05_008_S2_VLE_F4_F940_A3531', 'AMC_05_NDBE_F465_342mm_2300', 'AMC_04_HGD_F545_303mm_2130', 'AMC_03_LGD_F990_361mm_1630','AMC_02_NDBE_F630_361mm_1900', 'AMC_01_NDBE_F803_370mm_1800'] 
    print("Starting extracting the patches.\n-------------------------------------------------------\n")
    start = time.time()
    if version == 'prn':
        commands = [
            './darknet detector test data/obj12.data cfg/yolo-prn.cfg weights/prn/prn12/yolo-obj_4000.weights -thresh 0.15 -dont_show -ext_output < data/test12.txt > result.txt 2>/dev/null',
            './darknet detector test data/obj34.data cfg/yolo-prn.cfg weights/prn/prn34/yolo-obj_4000.weights -thresh 0.15 -dont_show -ext_output < data/test34.txt > result.txt 2>/dev/null',
            './darknet detector test data/obj56.data cfg/yolo-prn.cfg weights/prn/prn56/yolo-obj_4000.weights -thresh 0.15 -dont_show -ext_output < data/test56.txt > result.txt 2>/dev/null',
            './darknet detector test data/obj78.data cfg/yolo-prn.cfg weights/prn/prn78/yolo-obj_4000.weights -thresh 0.15 -dont_show -ext_output < data/test78.txt > result.txt 2>/dev/null',
            './darknet detector test data/obj910.data cfg/yolo-prn.cfg weights/prn/prn910/yolo-obj_4000.weights -thresh 0.15 -dont_show -ext_output < data/test910.txt > result.txt 2>/dev/null']
        temp = 4


    elif version == 'pan':
        commands = [
            './darknet detector test data/obj12.data cfg/yolo-pan.cfg weights/pan/pan12/yolo-obj_4000.weights -thresh 0.12 -dont_show -ext_output < data/test12.txt > result.txt',
            './darknet detector test data/obj34.data cfg/yolo-pan.cfg weights/pan/pan34/yolo-obj_4000.weights -thresh 0.12 -dont_show -ext_output < data/test34.txt > result.txt 2>/dev/null',
            './darknet detector test data/obj56.data cfg/yolo-pan.cfg weights/pan/pan56/yolo-obj_4000.weights -thresh 0.12 -dont_show -ext_output < data/test56.txt > result.txt 2>/dev/null',
            './darknet detector test data/obj78.data cfg/yolo-pan.cfg weights/pan/pan78/yolo-obj_4000.weights -thresh 0.12 -dont_show -ext_output < data/test78.txt > result.txt 2>/dev/null',
            './darknet detector test data/obj910.data cfg/yolo-pan.cfg weights/pan/pan910/yolo-obj_4000.weights -thresh 0.12 -dont_show -ext_output < data/test910.txt > result.txt 2>/dev/null']
        temp = 4

    elif version == 'yolov4':
        commands =[ \
        './darknet detector test data/obj12.data cfg/yolo-obj.cfg weights/yolov4/12/yolo-obj_4000.weights -thresh 0.25 -dont_show -ext_output < data/test12.txt > result.txt 2>/dev/null', \
        './darknet detector test data/obj34.data cfg/yolo-obj.cfg weights/yolov4/34/yolo-obj_4000.weights -thresh 0.25 -dont_show -ext_output < data/test34.txt > result.txt 2>/dev/null', \
        './darknet detector test data/obj56.data cfg/yolo-obj.cfg weights/yolov4/56/yolo-obj_4000.weights -thresh 0.25 -dont_show -ext_output < data/test56.txt > result.txt 2>/dev/null', \
        './darknet detector test data/obj78.data cfg/yolo-obj.cfg weights/yolov4/78/yolo-obj_4000.weights -thresh 0.25 -dont_show -ext_output < data/test78.txt > result.txt 2>/dev/null', \
        './darknet detector test data/obj910.data cfg/yolo-obj.cfg weights/yolov4/910/yolo-obj_4000.weights -thresh 0.25 -dont_show -ext_output < data/test910.txt > result.txt 2>/dev/null']
        temp = 6

    for iterate in range(len(commands)):
        print("Extracting tif stacks nr. " + str(2*iterate+1) +
              " and " + str(2*iterate+2) + "...")
        ex_starttime = time.time()
        os.system(commands[iterate])
        test_Path = 'result.txt'
        with open((test_Path), 'r') as fobj:
            for line in fobj:
                image_List = [[num for num in line.split()] for line in fobj]
                #print(image_List)
                image_List = image_List[temp:-1]
                
                
        img_num = -1
        x = [[[[0 for l in range(0)] for i in range(20)]
              for j in range(0, 51)] for z in range(2)]
        img_list = []
        batch_num = 0
        for i in image_List:
            if(i[0][0] == 'd'):
                detection_num = 0

                img_num = img_num + 1
                if(img_num > 50):
                    img_num = 0
                    batch_num = batch_num + 1
                img_list.append(i[0][:-1])

            if(i[0] == 'ROI:'):
                x[batch_num][img_num][detection_num].append(i[3])
                x[batch_num][img_num][detection_num].append(i[5])
                x[batch_num][img_num][detection_num].append(i[7])
                x[batch_num][img_num][detection_num].append(i[9][:-1])
                detection_num = detection_num + 1

        x = [[[[int(i) for i in thing] for thing in a if thing != []]
              for a in b] for b in x]  # remove empty lists

        #multiplier = 1.5  # put how much of the bouding box height you want

        path = os.getcwd()

        for it in range(2):
            for iterator in range(51):
                
                if os.path.exists(img_list[it*51 + iterator][:-4]):
                    shutil.rmtree(img_list[it*51 + iterator][:-4])
                os.mkdir(img_list[it*51 + iterator][:-4])
                img = Image.open(img_list[it*51 + iterator])
                os.chdir(img_list[it*51 + iterator][:-4])
                num = 1

                for i in range(len(x[it][iterator])):
                    if (x[it][iterator][i][2] > 50 and x[it][iterator][i][3] > 50):
                        img2 = img.crop((x[it][iterator][i][0], x[it][iterator][i][1], x[it][iterator][i]
                                         [0]+x[it][iterator][i][2], x[it][iterator][i][1]+x[it][iterator][i][3]*multiplier))
                        img2.save(img_list[it*51 + iterator]
                                  [9:-4] + "_" + str(num) + '.jpg')
                        num = num + 1
                os.chdir(path)

        print("Extraction time of tif stack " + str(2*iterate+1) + " and " + str(2 *
                                                                                 iterate+2) + ": " + str(round(time.time()-ex_starttime, 2)) + " seconds\n")
    print("Time to extract all predictions: " +
          str(round(time.time()-start, 2)) + " seconds")
    
    if os.path.exists('extracted_patches'):
        shutil.rmtree('extracted_patches')
    os.mkdir('extracted_patches')

    #os.system("mv ./data/obj/*/ extracted_patches")


    for i in range(len(names)):
        if os.path.exists("output_patches/" + names[i]):
            shutil.rmtree("output_patches/" + names[i])
        os.mkdir("output_patches/" + names[i])

        os.system("mv ./data/obj/*" + names[i] + "*/" +" " + "output_patches/" + names[i])



def tif_to_jpg(names):
    print("Starting the conversion of TIF stacks to JPG files.\n-------------------------------------------------------\n")
    #names=["GEN2LFIM_16_Cancer_F6_F489_A1252", 'GEN2LFIM_13_Cancer_F3_F569_A376', 'DDP_12_002_Squamous_F5_F955_A3128','DDP_10_005_NS2_VLE_F5_F642_A2173', 'DDP_05_008_S2_VLE_F4_F940_A3531', 'AMC_05_NDBE_F465_342mm_2300', 'AMC_04_HGD_F545_303mm_2130', 'AMC_03_LGD_F990_361mm_1630','AMC_02_NDBE_F630_361mm_1900', 'AMC_01_NDBE_F803_370mm_1800'] 
    start = time.time()
    im = np.zeros((len(names),), dtype=np.ndarray)
    for z in range(1, len(names)+1):

        file_path = 'tif_stacks/' + str(names[z-1]) + '.tif'
        dataset = Image.open(file_path)
        h, w = np.shape(dataset)
        tiffarray = np.zeros((h, w, dataset.n_frames))
        for i in range(dataset.n_frames):
            dataset.seek(i)
            tiffarray[:, :, i] = np.array(dataset)
        im[z-1] = tiffarray.astype(np.double)
    point1 = time.time()
    #print("Time to read the TIF data : " +
    #      str(round(point1 - start, 2)) + " seconds\n")
    # os.mkdir('data')
    # os.chdir('data')
    for i in range(0, len(names)):
        for iterator in range(0, 51):
            img = Image.fromarray(im[i][:, :, iterator]).convert("RGB")
            img.save("data/obj/" + names[i] +
                     "_frame_" + str(iterator+1) + '.jpg')
    end = time.time()
    #print("Time to convert the TIF stack data to JPG images: " +
    #      str(round(end - point1, 2)) + " seconds\n")
    print("Total time of TIF to JPG conversion : " +
          str(round(end-start, 2)) + " seconds\n")


def crop(names):
    print("Starting cropping JPG files to 682x682.\n-------------------------------------------------------\n")
    #names=["GEN2LFIM_16_Cancer_F6_F489_A1252", 'GEN2LFIM_13_Cancer_F3_F569_A376', 'DDP_12_002_Squamous_F5_F955_A3128','DDP_10_005_NS2_VLE_F5_F642_A2173', 'DDP_05_008_S2_VLE_F4_F940_A3531', 'AMC_05_NDBE_F465_342mm_2300', 'AMC_04_HGD_F545_303mm_2130', 'AMC_03_LGD_F990_361mm_1630','AMC_02_NDBE_F630_361mm_1900', 'AMC_01_NDBE_F803_370mm_1800'] 

    height = 682
    thresh_up = 80

    start = time.time()

    for num in range(0, len(names)):
        for iteri in range(0, 51):
            fname = "data/obj/" + names[num] + "_frame_" + str(iteri+1) + '.jpg'
            fname2 = "data/obj/batch" + str(num+1) + "_" + str(iteri+1)+".jpg"
            img = Image.open(fname).convert("L")
            arr = np.asarray(img)
            ballon = min(arr.argmax(0))
            img2 = arr[ballon-thresh_up:ballon-thresh_up+height, :]
            img3 = Image.fromarray(img2).convert("L")

            img3.save(fname)
    print("Time it took to crop images: " +
          str(round(time.time() - start, 2)) + " seconds\n")


if __name__ == "__main__":
    main()
