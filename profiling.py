import csv

def read_profile_csv(filename):
    """
    读取profile.csv文件并打印每个chip执行每种task的时间
    
    参数:
        filename: CSV文件路径
    """
    try:
        with open(filename, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            for row in reader:
                chip = row.get('chip', '').strip()
                task_type = row.get('task_type', '').strip()
                time = row.get('time', '').strip()
                
                # 跳过空行
                if not chip or not task_type or not time:
                    continue
                
                # 打印信息
                print(f"profile {chip} executed {task_type} in {time} seconds")
                
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
    except Exception as e:
        print(f"Error reading file: {str(e)}")

if __name__ == '__main__':
    filename = 'profile.csv'
    read_profile_csv(filename)

