import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-txt', type=str,default='coco/train2017.txt', help='train txt to check')
opt = parser.parse_args()
print(opt)

if __name__ == '__main__':
    traintxt = opt.txt
    with open(traintxt,'r') as f:
        lines = f.read().splitlines()

    img_files = lines
    label_files = [x.replace('images', 'labels').replace(os.path.splitext(x)[-1], '.txt') for x in img_files] 
    
    newtxt = traintxt+'.new'
    f_new=open(newtxt,'w')

    no_label_num = 0
    no_img_num = 0
    for i in range(len(img_files)):
        imgfile = img_files[i]
        labelfile = label_files[i]

        if os.path.isfile(imgfile):
            if os.path.isfile(labelfile):
                f_new.writelines(imgfile + '\n')
            else:
                no_label_num +=1
                print('{} does not exist'.format(labelfile))
        else:
            no_img_num +=1
            print('{} does not exist'.format(imgfile))
    print('{} imgs not exist'.format(no_img_num))
    print('{} imgs has no label'.format(no_label_num))
    f_new.close()