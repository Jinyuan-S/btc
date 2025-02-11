import pymongo as db
from math import *
import numpy as np
import matplotlib.pyplot as plt

MAX_HEIGHT = 508000

height_to_year = {32621: '2010', 100591: '2011', 160190: '2012', 214725: '2013', 278201: '2014', 336877: '2015', 391183: '2016', 446033: '2017', 501000: '2018'}

# start mongod client / 启动mongod client
client = db.MongoClient()
snapshot = client['btc']['snap']

height = []
total_value = []
p2pk_value = []
revealed_value = []
accounted_value = []
pot_revealed_value = []


docs = snapshot.find({})
for doc in docs:
    h = doc['_id']
    if h > MAX_HEIGHT:
        break
    height.append(h)
    # convert to bitcoin / 转换成比特币
    tot = float(doc['tot-val'])*1e-8*1e-3
    total_value.append(tot)
    lost = doc['unknown-frac']*tot
    accounted_value.append(tot-lost)
    finalp2pk = 0.0
    finalp2pkmulti = 0.0
    finalp2pkcomp = 0.0
    valp2pk = 0
    valp2sh = 0
    valp2shu = 0
    for item in doc['summary-by-type']:
        if item['type'] == 'P2PK':
            finalp2pk = float(item['tot-val'])*1e-8*1e-3
            valp2pk += finalp2pk
        if item['type'] == 'P2PK multisig':
            finalp2pkmulti = float(item['tot-val'])*1e-8*1e-3
            valp2pk += finalp2pkmulti
        if item['type'] == 'P2PK comp':
            finalp2pkcomp = float(item['tot-val'])*1e-8*1e-3
            valp2pk += finalp2pkcomp
        if item['type'] == 'P2PKH':
            finalp2pkh = float(item['tot-val'])*1e-8*1e-3
        if item['type'] == 'P2SH':
            valp2sh += float(item['tot-val'])*1e-8*1e-3
        if item['type'] == 'P2SH unknown':
            valp2shu += float(item['tot-val'])*1e-8*1e-3
    p2pk_value.append(valp2pk)
    qval = doc['qattack-frac']*tot
    revealed_value.append(qval)
    pot_revealed_value.append(qval + valp2shu)
    
plt.figure(figsize=(7, 5))
ax = plt.subplot(111) 
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.xlim((0, MAX_HEIGHT+5000))
plt.ylim((0, 17000))
plt.plot(height, total_value, 'C0-')
plt.plot(height, p2pk_value, 'C3--')
plt.plot(height, pot_revealed_value, 'C1--')
plt.plot(height, revealed_value, 'C3-')
plt.fill_between(height, 0, revealed_value, facecolor='red', alpha=0.05)
plt.fill_between(height, 0, pot_revealed_value, facecolor='red', alpha=0.05)
plt.axes

plt.plot(height[-1], revealed_value[-1], 'C3o')
plt.annotate("%02d%% of market cap" % int(qval/tot*100), 
        xy=(height[-1], revealed_value[-1]), xycoords='data',
        xytext=(380000, 7000), textcoords='data', 
        arrowprops=dict(facecolor='black', arrowstyle="->"))

plt.legend(['in circulation', 'at P2PK/multisig address', 'P2PK or address reused', 'PK known'])
plt.title('History of quantum vulnerable bitcoins')  # 易受量子攻击比特币的历史
#plt.xlabel('year')
plt.xticks(list(height_to_year.keys()), list(height_to_year.values()))
plt.ylabel('in kBTC')
plt.savefig('fig/address-history.pdf')
plt.savefig('fig/address-history.png')



plt.figure(figsize=(7, 5))
# Data to plot
labels = ['P2PK (including multisig and compressed)', 'P2PKH unlocked', 'P2SH unlocked', 'P2PKH hidden', 'P2SH hidden', 'other']
sizes = [finalp2pk + finalp2pkmulti + finalp2pkcomp, qval - valp2pk, valp2shu, finalp2pkh - qval + valp2pk, valp2sh, lost]
#colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
explode = (0.1, 0.1, 0.1, 0, 0, 0)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels,
        autopct='%1.1f%%', shadow=True, startangle=140)
 
plt.axis('equal')

plt.savefig('fig/address-summary.pdf')
plt.savefig('fig/address-summary.png')
