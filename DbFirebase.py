from firebase import firebase
from datetime import datetime

class DbFirebase:
    def __init__(self, url = 'https://rpi2-eb37b.firebaseio.com/', leaves= 0, flowers= 0,melons= 0):
        self.url = url
        self.leavers= leaves 
        self.flowers = flowers
        self.melons = melons
        
    def add(self,one = 'leaves', two='flowers', three='melons'):
        f = firebase.FirebaseApplication(self.url, None)
        dt_string = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
        result = f.patch('melon/'+str(dt_string), {one:self.leavers, two:self.flowers, three:self.melons})
        print (result)
        
    def showAll(self, key =False, row ='', column=''):
        f = firebase.FirebaseApplication(self.url, None)
        result = f.get('melon/'+row, column)
        if key:
            print (result.keys())
        else:
            print(result)
            
    def edit(self,one = None, two=None, three=None, row=''):
        f = firebase.FirebaseApplication(self.url, None)
        # ---
        if one == None and two ==None and three==None:
            print('please fill parameters at least 1 parameter')
        #+++
        elif one != None and two !=None and three!=None:
            result = f.patch('melon/'+row, {'flowers':one, 'leaves':two, 'melons':three})
        #+--
        elif one != None and two ==None and three==None:
            result = f.patch('melon/'+row, {'flowers':one})
        #-+-
        elif one == None and two !=None and three==None:
            result = f.patch('melon/'+row, {'leaves':two})
        #--+
        elif one == None and two ==None and three!=None:
            result = f.patch('melon/'+row, {'melons':three})
        #++-
        elif one != None and two !=None and three==None:
            result = f.patch('melon/'+row, {'flowers':one, 'leaves':two})
        #-++
        elif one == None and two !=None and three!=None:
            result = f.patch('melon/'+row, {'leaves':two, 'melons':three})
        #+-+
        else:
            result = f.patch('melon/'+row, {'flowers':one,'melons':three})
        print (result)
        
    def delete(self, row=None):
        f = firebase.FirebaseApplication(self.url, None)
        if row is None:
            print('cannot delete...please tell us the record you wan to delete!!!')
        else:
            f.delete('melon/'+ row, '')
            print('Record Deleted')
