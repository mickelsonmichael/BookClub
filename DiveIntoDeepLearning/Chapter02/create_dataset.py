import os

filename=os.path.join('..', 'data', 'house_tiny.csv')

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = filename
with open(data_file, 'w') as f:
    f.write('''NumRooms,RoofType,Price
NA,NA,127500
2,NA,106000
4,Slate,178100
NA,NA,140000''')

print('created', filename)