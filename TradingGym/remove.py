import os
elements = os.listdir('.')

for el in elements:
    if el.startswith("TradingGym"):
        if int(el[11:13])%1000==0:
            continue
        else:
            os.remove(el)
