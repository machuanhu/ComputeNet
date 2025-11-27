import json
import hashlib

class Identification:
    def __init__(self,location,industry,ownership,resource_type,node_info,chip_info,area,other_info):
        self.location = location
        self.industry = industry
        self.ownership = ownership
        self.resource_type = resource_type
        self.node_info = node_info
        self.chip_info = chip_info
        self.area = area
        self.other_info = other_info
        
    def show_attributes(self):
        for key, value in vars(self).items():
            print(f"{key}: {value}")

    def to_string(self):
        return '.'.join([f"{value}" for key, value in vars(self).items()])
    
class Mapper:
    def __init__(self):
        self.address_table = self.get_address_table()
    
    def get_address_table(self):
        file_path = "address_mapping.json"
        with open(file_path, "r") as json_file:
            data = json.load(json_file)
        return data
    
    def get_address(self,idt):
        return self.address_table[get_hash(idt.to_string())]

def generate_mapping():
    idt_list = get_identifications()
    idt_to_address = {}
    idt_to_address[get_hash(idt_list[0].to_string())] = '192.168.192.175'
    idt_to_address[get_hash(idt_list[1].to_string())] = '192.168.192.225'
    idt_to_address[get_hash(idt_list[2].to_string())] = '192.168.192.158'
    print(idt_to_address)

    hash_file_path = "address_mapping.json"
    with open(hash_file_path, "w") as json_file:
        json.dump(idt_to_address, json_file, indent=4)

def get_identifications():
    file_path = "identifications.json"
    with open(file_path, "r") as json_file:
        data = json.load(json_file)

    idt_list = []
    for idt_data in data:
        idt = Identification(**idt_data)
        idt_list.append(idt)
    return idt_list

def get_hash(value):
    return hashlib.sha256(value.encode()).hexdigest()

if __name__ == '__main__':
    generate_mapping()