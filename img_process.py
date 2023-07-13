import random
import cv2
import os


def create_lpsrnx(multiplier):
    super_lpsrn = cv2.dnn_superres.DnnSuperResImpl_create()
    super_lpsrn.readModel('models\lapSRN\LapSRN_x'+str(multiplier)+'.pb')
    super_lpsrn.setModel('lapsrn', multiplier)
    return super_lpsrn


def create_edsrx(multiplier):
    super_edsr = cv2.dnn_superres.DnnSuperResImpl_create()
    super_edsr.readModel('models\edsr\EDSR_x'+str(multiplier)+'.pb')
    super_edsr.setModel('edsr', multiplier)
    return super_edsr


def Predict(img):
    super_lpsrn2x = create_lpsrnx(2)
    super_lpsrn4x = create_lpsrnx(4)
    super_lpsrn8x = create_lpsrnx(8)

    super_edsr2x = create_edsrx(2)
    super_edsr4x = create_edsrx(4)
    imgs = {}
    # imgs['lpsrnx2'] = super_lpsrn2x.upsample(img)
    # print("generated lpsrnx2")
    # imgs['lpsrnx4'] = super_lpsrn4x.upsample(img)
    # print("generated lpsrnx4")
    # imgs['lpsrnx8'] = super_lpsrn8x.upsample(img)
    # print("generated lpsrnx8")
    # imgs['edsrx2'] = super_edsr2x.upsample(img)
    # print("generated edsr2")
    imgs['edsrx4'] = super_edsr4x.upsample(img)
    print("generated edsr4")
    imgs['original'] = img
    return imgs


def Save_temp(imgs):
    os.makedirs('./temp', exist_ok=True)
    paths = []
    random_number = random.randint(0, 1000000)
    path_c = "./temp/"+str(random_number)
    os.makedirs(path_c, exist_ok=True)
    for key in imgs.keys():
        path = './temp/{}/{}.png'.format(random_number, key)
        save_path = '{}/{}.png'.format(random_number, key)
        paths.append(save_path)
        result = cv2.imwrite(path, imgs[key])
        print("saving {}.png on path ./temp/{}, result: {}".format(key, path_c, result))
    return paths
