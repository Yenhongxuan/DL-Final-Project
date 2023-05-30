import os



from PIL import Image


root = './data2'
classes = os.listdir(root)
for c in classes:
    print(c)
    file_list = os.listdir(os.path.join(root, c))
    print(len(file_list))
    
    
    if not os.path.isdir(os.path.join(root, c, 'modified')):
        os.mkdir(os.path.join(root, c, 'modified'))
    
    
    counter = 1
    for file in file_list:
        if os.path.isdir(os.path.join(root, c, file)):
            continue
        name = c + '_{}'.format(counter)
        img = Image.open(os.path.join(root, c, file)).convert('RGB')
        img.save(os.path.join(root, c, 'modified', '{}.jpg'.format(name)))        
        counter += 1

