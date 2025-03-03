import time

class Stat:
    static_id = 0
    def __init__(self):
        Stat.static_id = 1
    
    def run(self):
        Stat.static_id = Stat.static_id + 1
        Stat.static_id = Stat.static_id % 8
        print(f"static_id={Stat.static_id}")        
        return True
        
def main():
    m = Stat()
    while True:
        m.run()
        time.sleep(1)

main()
        