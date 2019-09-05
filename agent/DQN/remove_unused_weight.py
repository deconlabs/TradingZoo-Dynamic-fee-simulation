import os

#delete statedict_path  = "./saves/Original/1/TradingGym_Rainbow_1000.pth"
save_location = "saves/Original"
erase_list = [2000,3000,4000]
for agent_num in range(1,31):
    for erase in erase_list:
        try:
            os.remove(os.path.join(save_location,str(agent_num),f"TradingGym_Rainbow_{erase}.pth"))
        except:
            print(f"{erase} doesn't exist")