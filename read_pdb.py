import numpy as np
import os
import random
from network import example_network
from keras import optimizers
import heapq


def read_pdb(filename):
    with open(filename, 'r') as file:
        strline_L = file.readlines()
        # print(strline_L)

    X_list = list()
    Y_list = list()
    Z_list = list()
    atomtype_list = list()
    for strline in strline_L:
        # removes all whitespace at the start and end, including spaces, tabs, newlines and carriage returns
        stripped_line = strline.strip()

        line_length = len(stripped_line)

        # print("Line length:{}".format(line_length))
        # if line_length != 78:
        #     print(filename)
        #     print("ERROR: line length is different. Expected=78, current={}".format(line_length))

        X_list.append(float(stripped_line[30:38].strip()) / 20)
        Y_list.append(float(stripped_line[38:46].strip()) / 20)
        Z_list.append(float(stripped_line[46:54].strip()) / 20)

        atomtype = stripped_line[76:78].strip()
        if atomtype == 'C':
            atomtype_list.append('h') # 'h' means hydrophobic
        else:
            atomtype_list.append('p') # 'p' means polar

    file.close()
    return X_list, Y_list, Z_list, atomtype_list


# [0, 0, 0, 0] --> [pro, lig, h, p]
def temp_add(temp, X_list, Y_list, Z_list, atomtype_list, flag):
    for j in range(len(X_list)):

        k = round(X_list[j] + 12.22005)
        m = round(Y_list[j] + 9.32035)
        n = round(Z_list[j] + 8.8514)

        if flag == 1:
            if atomtype_list[j] == 'h':
                if temp[k][m][n][1] == 1:
                    temp[k][m][n] = [1, 1, 1, 0]
                    if temp[k][m][n][3] == 1:
                        temp[k][m][n] = [1, 1, 1, 1]
                elif temp[k][m][n][3] == 1:
                    temp[k][m][n] = [1, 0, 1, 1]
                else:
                    temp[k][m][n] = [1, 0, 1, 0]
                # temp[k][m][n] += np.array([1, 0, 1, 0])
            else:
                if temp[k][m][n][1] == 1:
                    temp[k][m][n] = [1, 1, 0, 1]
                    if temp[k][m][n][2] == 1:
                        temp[k][m][n] = [1, 1, 1, 1]
                elif temp[k][m][n][2] == 1:
                    temp[k][m][n] = [1, 0, 1, 1]
                else:
                    temp[k][m][n] = [1, 0, 0, 1]
                # temp[k][m][n] += np.array([1, 0, 0, 1])
        else:
            if atomtype_list[j] == 'h':
                if temp[k][m][n][0] == 1:
                    temp[k][m][n] = [1, 1, 1, 0]
                    if temp[k][m][n][3] == 1:
                        temp[k][m][n] = [1, 1, 1, 1]
                elif temp[k][m][n][3] == 1:
                    temp[k][m][n] = [0, 1, 1, 1]
                else:
                    temp[k][m][n] = [0, 1, 1, 0]
                # temp[k][m][n] += np.array([0, 1, 1, 0])
            else:
                if temp[k][m][n][0] == 1:
                    temp[k][m][n] = [1, 1, 0, 1]
                    if temp[k][m][n][2] == 1:
                        temp[k][m][n] = [1, 1, 1, 1]
                elif temp[k][m][n][2] == 1:
                    temp[k][m][n] = [0, 1, 1, 1]
                else:
                    temp[k][m][n] = [0, 1, 0, 1]
                # temp[k][m][n] += np.array([0, 1, 0, 1])


def one_train_x(subject_path, corr_legend):
    temp = np.zeros((32, 32, 32, 4), dtype=np.float32)
    X_list, Y_list, Z_list, atomtype_list = read_pdb(subject_path)
    X_list_lig, Y_list_lig, Z_list_lig, atomtype_list_lig = read_pdb(corr_legend)
    temp_add(temp, X_list, Y_list, Z_list, atomtype_list, 1)
    temp_add(temp, X_list_lig, Y_list_lig, Z_list_lig, atomtype_list_lig, 0)
    return temp


def get_random(up_limit, value_limit):
    rom_legend_file_num = random.randint(1, up_limit)
    while rom_legend_file_num == value_limit:
        rom_legend_file_num = random.randint(1, up_limit)

    if rom_legend_file_num >= 1000:
        return str(rom_legend_file_num)
    if rom_legend_file_num >= 100:
        return '0' + str(rom_legend_file_num)
    if rom_legend_file_num >= 10:
        return '00' + str(rom_legend_file_num)
    return '000' + str(rom_legend_file_num)


def dataset_read(foldname):
    train_x, train_y = [], []
    test_x, test_y =[], []

    for dirname, dirnames, filenames in os.walk(foldname):
        for filename in filenames:
            subject_path = os.path.join(dirname, filename)
            if subject_path == os.path.join(dirname, '.DS_Store'):
                continue
            if subject_path[19:22] == 'lig':
                continue

            num = 2

            if int(subject_path[14:18]) < 2700:
                tem_name = subject_path[14:18] + '_lig_cg.pdb'
                corr_legend = os.path.join(subject_path[0:14], tem_name)
                temp_r = one_train_x(subject_path, corr_legend)
                train_x.append(temp_r)
                train_y.append([1])

                for i in range(num):
                    rom_legend_file_num = get_random(3000, int(subject_path[14:18]))
                    tem_name = str(rom_legend_file_num) + '_lig_cg.pdb'
                    corr_legend = os.path.join(subject_path[0:14], tem_name)
                    temp_f = one_train_x(subject_path, corr_legend)
                    train_x.append(temp_f)
                    train_y.append([0])

            else:
                tem_name = subject_path[14:18] + '_lig_cg.pdb'
                corr_legend = os.path.join(subject_path[0:14], tem_name)
                temp_r = one_train_x(subject_path, corr_legend)
                test_x.append(temp_r)
                test_y.append([1])

                num = 3000 - 2699

                for i in range(num):
                    rom_legend_file_num = 2700 + i
                    if rom_legend_file_num == int(subject_path[14:18]):
                        continue
                    tem_name = str(rom_legend_file_num) + '_lig_cg.pdb'
                    corr_legend = os.path.join(subject_path[0:14], tem_name)
                    temp_f = one_train_x(subject_path, corr_legend)
                    test_x.append(temp_f)

                for i in range(num - 1):
                    test_y.append([0])

            # count = 0
            # for k in range(32):
            #     for m in range(32):
            #         for n in range(32):
            #             if temp[k][m][n].any():
            #                 count += 1
            # print(count)

            # for tmp in X_list:
            #     x.append(tmp)
            # for tmp in Y_list:
            #     y.append(tmp)
            # for tmp in Z_list:
            #     z.append(tmp)

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    print(train_x.shape)
    print(train_y.shape)
    print(test_x.shape)
    print(test_y.shape)
    return train_x, train_y, test_x, test_y

    # x_array = np.array(x)
    # y_array = np.array(y)
    # z_array = np.array(z)
    # print(x_array.max(), x_array.min(), x_array.max() - x_array.min(), x_array.shape)
    # print(y_array.max(), y_array.min(), y_array.max() - y_array.min(), y_array.shape)
    # print(z_array.max(), z_array.min(), z_array.max() - z_array.min(), z_array.shape)


train_x, train_y, test_x, test_y = dataset_read("training_data")


model = example_network((32, 32, 32, 4))
adam = optimizers.Adam(lr=0.0001)
model.compile(optimizer=adam,
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(x=train_x, y=train_y, epochs=50, verbose=1, batch_size=15)

model.save('model.h5')

loss, acc = model.evaluate(x=test_x, y=test_y)
print(loss, acc)

num_pro = 3000 - 2699
num_lig = 3000 - 2699

count = 0

for i in range(num_pro):
    out = []
    for j in range(num_lig):
        pre = model.predict(np.expand_dims(test_x[i*num_lig+j], axis=0))
        out.append(pre[0])
    out = np.array(out).reshape((1,len(out)))[0]
    out = heapq.nlargest(10, range(len(out)), out.take)
    # print(out)
    if 0 in out:
        count += 1

print(count)
print(count / num_lig)

# 310.935 -244.401 555.336 (2795377,)
# 432.956 -186.407 619.363 (2795377,)
# 345.222 -177.028 522.25 (2795377,)
