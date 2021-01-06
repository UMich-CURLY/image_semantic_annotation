import os
path1 = '/home/luoxin/Documents/cheetah/dataset_merged/wave_field-182/look_down/labels_rgb/'
path2 = '/home/luoxin/Documents/cheetah/dataset_merged/wave_field-182/look_down/labels_id/'
path3 = '/home/luoxin/Documents/cheetah/dataset_merged/wave_field-182/look_down/images/'

filename_list1 = os.listdir(path1)
filename_list1.sort()
filename_list2 = os.listdir(path2)
filename_list2.sort()
filename_list3 = os.listdir(path3)
filename_list3.sort()

a = 0
b = 5831
for i in filename_list1:
	used_name = path1 + filename_list1[a]
	new_name = path1 + 'image' + str(b) + '.png'
	os.rename(used_name, new_name)
	a += 1
	b += 1

a = 0
b = 5831
for i in filename_list2:
	used_name = path2 + filename_list2[a]
	new_name = path2 + 'image' + str(b) + '.png'
	os.rename(used_name, new_name)
	a += 1
	b += 1

a = 0
b = 5831
for i in filename_list3:
	used_name = path3 + filename_list3[a]
	new_name = path3 + 'image' + str(b) + '.png'
	os.rename(used_name, new_name)
	a += 1
	b += 1